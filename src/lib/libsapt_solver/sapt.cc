#include "sapt.h"

namespace psi { namespace sapt {

SAPT::SAPT(Options& options, boost::shared_ptr<PSIO> psio, 
  boost::shared_ptr<Chkpt> chkpt) : Wavefunction(options, psio, chkpt)
{
#ifdef HAVE_MKL
  mkl_set_dynamic(1);
#endif

#ifdef _OPENMP
  omp_set_nested(0);
#endif

  initialize();
  get_denom();
}

SAPT::~SAPT()
{
  if (evalsA_ != NULL) free(evalsA_);
  if (evalsB_ != NULL) free(evalsB_);
  if (diagAA_ != NULL) free(diagAA_);
  if (diagBB_ != NULL) free(diagBB_);
  if (CA_ != NULL) free_block(CA_);
  if (CB_ != NULL) free_block(CB_);
  if (CHFA_ != NULL) free_block(CHFA_);
  if (CHFB_ != NULL) free_block(CHFB_);
  if (sAB_ != NULL) free_block(sAB_);
  if (vABB_ != NULL) free_block(vABB_);
  if (vBAA_ != NULL) free_block(vBAA_);
  if (vAAB_ != NULL) free_block(vAAB_);
  if (vBAB_ != NULL) free_block(vBAB_);
  ribasis_.reset();
  zero_.reset();
}

void SAPT::initialize()
{
  evalsA_ = NULL;
  evalsB_ = NULL;
  diagAA_ = NULL;
  diagBB_ = NULL;
  CA_ = NULL;
  CB_ = NULL;
  CHFA_ = NULL;
  CHFB_ = NULL;
  sAB_ = NULL;
  vABB_ = NULL;
  vBAA_ = NULL;
  vAAB_ = NULL;
  vBAB_ = NULL;

  boost::shared_ptr<BasisSetParser> parser(new Gaussian94BasisSetParser());
  //
  // If the user doesn't spec a basis name, pick it yourself
  // TODO: Verify that the basis assign does not messs this up
  if (options_.get_str("RI_BASIS_SAPT") == "") {
    basisset_->molecule()->set_basis_all_atoms(options_.get_str("BASIS") 
      + "-RI", "RI_BASIS_SAPT");
    fprintf(outfile, "    No auxiliary basis selected, defaulting to %s-RI\n\n", options_.get_str("BASIS").c_str()); 
  }

  ribasis_ = boost::shared_ptr<BasisSet>(BasisSet::construct(parser, molecule_, 
    "RI_BASIS_SAPT"));
  elst_basis_ = 0;
  if (options_.get_str("RI_BASIS_ELST") != "") {
    elstbasis_ = boost::shared_ptr<BasisSet>(BasisSet::construct(parser, 
      molecule_,"RI_BASIS_ELST"));
    elst_basis_ = 1;
  }
  zero_ = boost::shared_ptr<BasisSet>(BasisSet::zero_ao_basis_set());
  parser.reset();


  print_ = options_.get_int("PRINT");
  debug_ = options_.get_int("DEBUG");
  schwarz_ = options_.get_double("SCHWARZ_CUTOFF");
  mem_ = (long int) ((double) memory_*options_.get_double("SAPT_MEM_SAFETY"));
  mem_ /= 8L;

  if (options_["NFRZ_A"].has_changed() || options_["NFRZ_B"].has_changed()) {
    foccA_ = options_.get_int("NFRZ_A");
    foccB_ = options_.get_int("NFRZ_B");
  }
  else {
    std::vector<int> realsA;
    realsA.push_back(0);
    std::vector<int> ghostsA;
    ghostsA.push_back(1);
    boost::shared_ptr<Molecule> monomerA = molecule_->extract_subsets(realsA,
      ghostsA);
    foccA_ = monomerA->nfrozen_core(options_.get_str("FREEZE_CORE"));

    std::vector<int> realsB;
    realsB.push_back(1);
    std::vector<int> ghostsB;
    ghostsB.push_back(0);
    boost::shared_ptr<Molecule> monomerB = molecule_->extract_subsets(realsB,
      ghostsB);
    foccB_ = monomerB->nfrozen_core(options_.get_str("FREEZE_CORE"));
  }

  ndf_ = ribasis_->nbf();

  psio_->open(PSIF_SAPT_DIMER,PSIO_OPEN_OLD);
  psio_->open(PSIF_SAPT_MONOMERA,PSIO_OPEN_OLD);
  psio_->open(PSIF_SAPT_MONOMERB,PSIO_OPEN_OLD);

  double enucD, enucA, enucB;
  double eHFD, eHFA, eHFB;

  psio_->read_entry(PSIF_SAPT_DIMER,"Dimer NSO",(char *) &nso_,sizeof(int));
  psio_->read_entry(PSIF_SAPT_DIMER,"Dimer NMO",(char *) &nmo_,sizeof(int));
  psio_->read_entry(PSIF_SAPT_DIMER,"Dimer HF Energy",(char *) &eHFD, 
    sizeof(double));
  psio_->read_entry(PSIF_SAPT_DIMER,"Dimer Nuclear Repulsion Energy",(char *)
    &enucD, sizeof(double));

  int nsotri = nso_*(nso_+1)/2;
  double *S = init_array(nsotri);
  psio_->read_entry(PSIF_SAPT_DIMER,"Dimer Overlap Integrals",(char *) &S[0], 
    sizeof(double)*nsotri);

  int nsoA, nmoA;

  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer NSO",(char *) &nsoA, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer NMO",(char *) &nmoA, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer NOCC",(char *) &noccA_, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer NVIR",(char *) &nvirA_, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer Number of Electrons",(char *)
    &NA_, sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer HF Energy",(char *) &eHFA, 
    sizeof(double));
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer Nuclear Repulsion Energy",
    (char *) &enucA, sizeof(double));

  aoccA_ = noccA_ - foccA_;

  if (nsoA != nso_) 
    throw PsiException("Number of orbitals do not match", __FILE__,
      __LINE__);
  if (nmoA != nmo_)
    throw PsiException("Number of orbitals do not match", __FILE__,
      __LINE__);

  int nsoB, nmoB;

  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer NSO",(char *) &nsoB, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer NMO",(char *) &nmoB, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer NOCC",(char *) &noccB_, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer NVIR",(char *) &nvirB_, 
    sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer Number of Electrons",(char *)
    &NB_, sizeof(int));
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer HF Energy",(char *) &eHFB, 
    sizeof(double));
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer Nuclear Repulsion Energy",
    (char *) &enucB, sizeof(double));

  aoccB_ = noccB_ - foccB_;

  if (nsoB != nso_)
    throw PsiException("Number of orbitals do not match", __FILE__,
      __LINE__);
  if (nmoB != nmo_)
    throw PsiException("Number of orbitals do not match", __FILE__,
      __LINE__);

  enuc_ = enucD - enucA - enucB;
  eHF_ =  eHFD - eHFA - eHFB;

  evalsA_ = init_array(nmo_);
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer HF Eigenvalues",(char *)
    &(evalsA_[0]), sizeof(double)*nmo_);

  double *VA = init_array(nsotri);
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer Nuclear Attraction Integrals",
    (char *) &(VA[0]), sizeof(double)*nsotri);

  CA_ = block_matrix(nso_,nmo_);
  psio_->read_entry(PSIF_SAPT_MONOMERA,"Monomer HF Coefficients",(char *)
    &(CA_[0][0]), sizeof(double)*nmo_*nso_);

  evalsB_ = init_array(nmo_);
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer HF Eigenvalues",(char *)
    &(evalsB_[0]), sizeof(double)*nmo_);

  double *VB = init_array(nsotri);
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer Nuclear Attraction Integrals",
    (char *) &(VB[0]), sizeof(double)*nsotri);

  CB_ = block_matrix(nso_,nmo_);
  psio_->read_entry(PSIF_SAPT_MONOMERB,"Monomer HF Coefficients",(char *)
    &(CB_[0][0]), sizeof(double)*nmo_*nso_);

  psio_->close(PSIF_SAPT_DIMER,1);
  psio_->close(PSIF_SAPT_MONOMERA,1);
  psio_->close(PSIF_SAPT_MONOMERB,1);

  double **sIJ = block_matrix(nso_,nso_);
  double **sAJ = block_matrix(nmo_,nso_);
  sAB_ = block_matrix(nmo_,nmo_);

  tri_to_sq(S,sIJ,nso_);

  C_DGEMM('T','N',nmo_,nso_,nso_,1.0,CA_[0],nmo_,sIJ[0],nso_,0.0,sAJ[0],nso_);
  C_DGEMM('N','N',nmo_,nmo_,nso_,1.0,sAJ[0],nso_,CB_[0],nmo_,0.0,sAB_[0],nmo_);

  free(S);
  free_block(sIJ);
  free_block(sAJ);

  double **vIJ = block_matrix(nso_,nso_);
  double **vIB = block_matrix(nso_,nmo_);
  double **vAJ = block_matrix(nmo_,nso_);
  vAAB_ = block_matrix(nmo_,nmo_);
  vABB_ = block_matrix(nmo_,nmo_);
  vBAA_ = block_matrix(nmo_,nmo_);
  vBAB_ = block_matrix(nmo_,nmo_);

  tri_to_sq(VA,vIJ,nso_);

  C_DGEMM('N','N',nso_,nmo_,nso_,1.0,vIJ[0],nso_,CB_[0],nmo_,0.0,
    vIB[0],nmo_);
  C_DGEMM('T','N',nmo_,nmo_,nso_,1.0,CA_[0],nmo_,vIB[0],nmo_,0.0,
    vAAB_[0],nmo_);
  C_DGEMM('T','N',nmo_,nmo_,nso_,1.0,CB_[0],nmo_,vIB[0],nmo_,0.0,
    vABB_[0],nmo_);

  tri_to_sq(VB,vIJ,nso_);

  C_DGEMM('T','N',nmo_,nso_,nso_,1.0,CA_[0],nmo_,vIJ[0],nso_,0.0,
    vAJ[0],nso_);
  C_DGEMM('N','N',nmo_,nmo_,nso_,1.0,vAJ[0],nso_,CA_[0],nmo_,0.0,
    vBAA_[0],nmo_);
  C_DGEMM('N','N',nmo_,nmo_,nso_,1.0,vAJ[0],nso_,CB_[0],nmo_,0.0,
    vBAB_[0],nmo_);

  free(VA);
  free(VB);
  free_block(vIJ);
  free_block(vIB);
  free_block(vAJ);
}

void SAPT::get_denom()
{
  boost::shared_ptr<Vector> evals_aoccA(new Vector(aoccA_));
  boost::shared_ptr<Vector> evals_virA(new Vector(nvirA_));
  boost::shared_ptr<Vector> evals_aoccB(new Vector(aoccB_));
  boost::shared_ptr<Vector> evals_virB(new Vector(nvirB_));

  for (int a=0; a<aoccA_; a++)
    evals_aoccA->set(0,a,evalsA_[a+foccA_]);
  for (int r=0; r<nvirA_; r++)
    evals_virA->set(0,r,evalsA_[r+noccA_]);
  for (int b=0; b<aoccB_; b++)
    evals_aoccB->set(0,b,evalsB_[b+foccB_]);
  for (int s=0; s<nvirB_; s++)
    evals_virB->set(0,s,evalsB_[s+noccB_]);

  denom_ = boost::shared_ptr<SAPTLaplaceDenominator>(
    new
SAPTLaplaceDenominator(evals_aoccA,evals_virA,evals_aoccB,evals_virB,
    options_.get_double("DENOMINATOR_DELTA"),debug_));

  boost::shared_ptr<Matrix> tauAR = denom_->denominatorA();
  boost::shared_ptr<Matrix> tauBS = denom_->denominatorB();

  dAR_ = tauAR->pointer();
  dBS_ = tauBS->pointer();

  nvec_ = denom_->nvector();
}

CPHFDIIS::CPHFDIIS(int length, int maxvec)
{
  max_diis_vecs_ = maxvec;
  vec_length_ = length;

  curr_vec_ = 0;
  num_vecs_ = 0;

  t_vecs_ = block_matrix(maxvec,length);
  err_vecs_ = block_matrix(maxvec,length);
}

CPHFDIIS::~CPHFDIIS()
{
  free_block(t_vecs_);
  free_block(err_vecs_);
}

void CPHFDIIS::store_vectors(double *t_vec, double *err_vec)
{
  C_DCOPY(vec_length_,t_vec,1,t_vecs_[curr_vec_],1);
  C_DCOPY(vec_length_,err_vec,1,err_vecs_[curr_vec_],1);

  curr_vec_ = (curr_vec_+1)%max_diis_vecs_;
  num_vecs_++;
  if (num_vecs_ > max_diis_vecs_) num_vecs_ = max_diis_vecs_;
}

void CPHFDIIS::get_new_vector(double *t_vec)
{
  int *ipiv;
  double *Cvec;
  double **Bmat;

  ipiv = init_int_array(num_vecs_+1);
  Bmat = block_matrix(num_vecs_+1,num_vecs_+1);
  Cvec = (double *) malloc((num_vecs_+1)*sizeof(double));

  for (int i=0; i<num_vecs_; i++) {
    for (int j=0; j<=i; j++) {
      Bmat[i][j] = Bmat[j][i] = C_DDOT(vec_length_,err_vecs_[i],1,
        err_vecs_[j],1);
  }}

  for (int i=0; i<num_vecs_; i++) {
    Bmat[num_vecs_][i] = -1.0;
    Bmat[i][num_vecs_] = -1.0;
    Cvec[i] = 0.0;
  }

  Bmat[num_vecs_][num_vecs_] = 0.0;
  Cvec[num_vecs_] = -1.0;

  C_DGESV(num_vecs_+1,1,&(Bmat[0][0]),num_vecs_+1,&(ipiv[0]),&(Cvec[0]),
    num_vecs_+1);

  for (int i=0; i<num_vecs_; i++) {
    C_DAXPY(vec_length_,Cvec[i],t_vecs_[i],1,t_vec,1);
  }

  free(ipiv);
  free(Cvec);
  free_block(Bmat);
}

}}
