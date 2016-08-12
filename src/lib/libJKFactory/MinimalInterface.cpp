/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "MinimalInterface.h"
#include "libmints/basisset_parser.h"
#include "libmints/basisset.h"
#include "libmints/matrix.h"
#include "libmints/wavefunction.h"
#include "libmints/molecule.h"
#include "libmints/twobody.h"
#include "libmints/integral.h"
#include "psi4-dec.h"
#include "../libparallel2/Communicator.h"
#include "../libparallel2/ParallelEnvironment.h"
#include "../libparallel2/Algorithms.h"
extern "C" {
   #include "CifiedFxns.h"
   #include "gtfock/libcint/CInt.h"
   #include "gtfock/pfock/pfock.h"
}


typedef boost::shared_ptr<psi::BasisSetParser> SharedParser;
typedef boost::shared_ptr<psi::BasisSet> SharedBasis;
typedef boost::shared_ptr<psi::Matrix> SharedMatrix;
typedef boost::shared_ptr<const psi::LibParallel::Communicator>
        ConstSharedComm;
typedef boost::shared_ptr<psi::LibParallel::Communicator>
        SharedComm;
//Helper fxns
void MakeBasis(BasisSet** GTBasis,SharedBasis PsiBasis);

typedef void (psi::MinimalInterface::*GetPutFxn)(
                         std::vector<SharedMatrix>& );


void psi::MinimalInterface::Vectorize(
      SharedMatrix Mat,GetPutFxn fxn){
   std::vector<SharedMatrix>temp(1,Mat);
   (this->*(fxn))(temp);
}

psi::MinimalInterface::~MinimalInterface(){
   PFock_destroy(PFock_);
   CInt_destroyBasisSet(GTBasis_);
   Comm_->FreeComm();
}

psi::MinimalInterface::MinimalInterface(const int NMats,
      const bool AreSymm):NPRow_(1),NPCol_(1),
                          StartRow_(0),StartCol_(0),
                          EndRow_(0),EndCol_(0),Stride_(0),NBasis_(0){
    //Trying to parallelize over the density matrices
    //Try to split up communicators
    psi::Options& options = psi::Process::environment.options;
    int density_matrices_per_process = options.get_int("DENSITY_MATRICES_PER_PROCESS");
    int subgroup = 1;
    if(density_matrices_per_process == 0 or NMats < 4)
    {
        std::string Comm_subgroup = "WorldComm";
        Comm_ = psi::WorldComm->GetComm()->MakeComm(1, Comm_subgroup);
        std::vector<int> density_list(NMats, 0);
        std::iota(density_list.begin(), density_list.end(), 0);
        subgroup_to_density_[subgroup] = density_list;
    }
    else {
        subgroup = NMats / density_matrices_per_process;
        int total_number_process = psi::WorldComm->GetComm()->NProc();
        int processor_per_group = total_number_process / subgroup;
        outfile->Printf("\n subgroup: %d total_number_process: %d processor_per_group: %d", subgroup, total_number_process, processor_per_group);
        int which_processor_group = (psi::WorldComm->GetComm()->Me() % subgroup);
        std::string Comm_subgroup = "SubDensComm" + std::to_string(which_processor_group);
        Comm_ = psi::WorldComm->GetComm()->MakeComm(which_processor_group, Comm_subgroup);
        for(int i = 0; i < subgroup; i++) {
            std::vector<int> density_list;
            if(i != subgroup - 1) 
            {
                density_list.resize(density_matrices_per_process);
                std::iota(density_list.begin(), density_list.end(), i * density_matrices_per_process);
            }
            else {
                density_list.resize(density_matrices_per_process + NMats % density_matrices_per_process);
                std::iota(density_list.begin(), density_list.end(), i * density_matrices_per_process);
            }
            subgroup_to_density_[subgroup] = density_list;

        }
    }
    SetUp();
    SplitProcs(NPRow_,NPCol_);
    outfile->Printf("\n Using GTFock for ParallelJK build with %d MPI processes and %d omp threads", NPRow_ * NPCol_, omp_get_max_threads());
    SharedBasis primary = psi::BasisSet::pyconstruct_orbital(
    		                  psi::Process::environment.legacy_molecule(),
                              "BASIS", options.get_str("BASIS"));
   NBasis_=primary->nbf();
   BlockDims(NBasis_);
   MakeBasis(&GTBasis_,primary);
   //I don't know how GTFock works
   double IntThresh=
         psi::Process::environment.options["INTS_TOLERANCE"].to_double();
   //It appears I can have GTFock figure this value out if I hand it
   //a negative value.
   int NBlkFock=-1;
   Timer pfock_create;
   int which_processor_group = (psi::WorldComm->GetComm()->Me() % (subgroup + 1));
   size_t my_group_size = subgroup_to_density_[subgroup].size();

   outfile->Printf("\n which_processor_group: %d my_group_size: %lu", which_processor_group, my_group_size);
   int return_flag = (int) PFock_create(GTBasis_,NPRow_,NPCol_,NBlkFock,IntThresh,
         static_cast<int>(my_group_size) ,AreSymm,&PFock_);

   if(return_flag != 0)
   {
       
       outfile->Printf("\n Something happened during PFock_create"); 
       outfile->Printf("\n GTFock error is %d", return_flag);
       outfile->Printf("\n NMats: %d AreSymm: %d", NMats, AreSymm);
       throw PSIEXCEPTION("GTFock threw a failure in PFock_create. Check error and log.");
   }
   outfile->Printf("\n PFock_create is complete.  Took %8.6f s. ", pfock_create.get());
}

void psi::MinimalInterface::SetP(std::vector<SharedMatrix>& Ps){
   int return_flag = 0;
   return_flag = (int)PFock_setNumDenMat(Ps.size(),PFock_);
   if(return_flag != 0)
   {
        outfile->Printf("\n PFock_setNumDenMat gave an error.");
        throw PSIEXCEPTION("PFock_setNumDenMat is FUBAR");
   }
   double* Buffer;
   for(int i=0;i<Ps.size();i++){
      MyBlock(&Buffer,Ps[i]);
      return_flag = (int)PFock_putDenMat(StartRow_,EndRow_,
                           StartCol_,EndCol_,Stride_,Buffer,i,PFock_);
   }
   return_flag = (int)PFock_commitDenMats(PFock_);
   return_flag = (int)PFock_computeFock(GTBasis_,PFock_);
   //PFock_getStatistics(PFock_);
   delete [] Buffer;
   if(return_flag != 0)
   {
        outfile->Printf("\n SetP (driver for GTFock failed");
        throw PSIEXCEPTION("PSI4 failed in SetP due to GTFock");
   }
}

void psi::MinimalInterface::GenGetCall(
      std::vector<SharedMatrix>& JorK,
      int value){
   int nrows=(EndRow_-StartRow_+1);
   int ncols=(EndCol_-StartCol_+1);
   double* Block=new double[nrows*ncols];
   for(int i=0;i<JorK.size();i++){
      memset(Block,0.0,sizeof(double)*nrows*ncols);
      int return_flag = (int) PFock_getMat(PFock_,(PFockMatType_t)value,i,StartRow_,
                  EndRow_,StartCol_,EndCol_,Stride_,Block);
      if(return_flag != 0)
      {
        outfile->Printf("\n PFock_getMat failed");
        throw PSIEXCEPTION("PSI4 failed in PFock_getMat due to GTFOCK");
      }
      Gather(JorK[i],Block);
   }
   delete [] Block;
}


void psi::MinimalInterface::GetJ(std::vector<SharedMatrix>& Js){
   GenGetCall(Js,(int)PFOCK_MAT_TYPE_J);
   for(int i=0;i<Js.size();i++)Js[i]->scale(0.5);
}

void psi::MinimalInterface::GetK(std::vector<SharedMatrix>& Ks){
   GenGetCall(Ks,(int)PFOCK_MAT_TYPE_K);
   for(int i=0;i<Ks.size();i++)Ks[i]->scale(-1.0);
}

void psi::MinimalInterface::MyBlock(double **Buffer,
                                    SharedMatrix Matrix){
   int nrows=EndRow_-StartRow_+1;
   (*Buffer)=new double[nrows*Stride_];
   for(int row=StartRow_;row<=EndRow_;++row){
      for(int col=StartCol_;col<=EndCol_;++col){
         (*Buffer)[(row-StartRow_)*Stride_+(col-StartCol_)]=
               (*Matrix)(row,col);
      }
   }
}

void FillMat(SharedMatrix Result, int RowStart, int RowEnd,
      int ColStart, int ColEnd, double* Buffer){
   int nrows=RowEnd-RowStart+1;
   int ncols=ColEnd-ColStart+1;
   for(int row=RowStart;row<=RowEnd;row++){
      for(int col=ColStart;col<=ColEnd;col++){
         (*Result)(row,col)=
               Buffer[(row-RowStart)*ncols+(col-ColStart)];
      }
   }
}


void psi::MinimalInterface::Gather(SharedMatrix Result,
                                   double *Block){
   if(!Result)Result=SharedMatrix(new psi::Matrix(NBasis_,NBasis_));
   SharedMatrix temp(new psi::Matrix(NBasis_,NBasis_));
   FillMat(temp,StartRow_,EndRow_,StartCol_,EndCol_,Block);
   //ConstSharedComm Comm=psi::WorldComm->GetComm();
   Comm_->AllReduce(&(*temp)(0,0),NBasis_*NBasis_,&(*Result)(0,0),LibParallel::ADD);
}

void psi::MinimalInterface::BlockDims(const int NBasis){
   //ConstSharedComm Comm=psi::WorldComm->GetComm();
   int MyRank=Comm_->Me();
   int IDs[2];
   //Divide the mat into NPRow_ by NPCol_ blocks
   //Note the following works because I know NPRow_*NPCol_=NProc
   IDs[0]=MyRank/NPCol_;//Row of my block
   IDs[1]=MyRank%NPCol_;//Col of my block
   for(int i=0;i<2;i++){
      SharedComm TempComm=Comm_->MakeComm(IDs[i]);
      int MyDim=TempComm->Me();
      int & DimStart=(i==0?StartRow_:StartCol_);
      int & DimEnd=(i==0?EndRow_:EndCol_);
      int & NP=(i==0?NPRow_:NPCol_);
      int BFsPerDimPerBlock=NBasis/NP;
      DimStart=IDs[i]*BFsPerDimPerBlock;
      //This needs to be the last actually usable value
      DimEnd=(IDs[i]==(NP-1)?NBasis:DimStart+BFsPerDimPerBlock)-1;
      TempComm->FreeComm();
   }
   Stride_=EndCol_-StartCol_+1;
}

void psi::MinimalInterface::SplitProcs(int& NPRow, int& NPCol){
   //ConstSharedComm Comm=psi::WorldComm->GetComm();
   int NProc=Comm_->NProc();
   NPRow=(int)floor(std::sqrt((double)NProc));
   bool done=false;
   while (!done) {
      if (NProc%NPRow==0) {
         NPCol=NProc/NPRow;
         done=true;
      }
      else NPRow--;
   }
}

void MakeBasis(BasisSet** GTBasis,SharedBasis PsiBasis){
   CInt_createBasisSet(GTBasis);
   boost::shared_ptr<psi::Molecule> PsiMol=PsiBasis->molecule();
   int NAtoms=PsiMol->natom();
   int NPrims=PsiBasis->nprimitive();
   int NShells=PsiBasis->nshell();
   int IsPure=(PsiBasis->has_puream()?1:0);
   //Carts,then atomic numbers
   std::vector<double> X(NAtoms),Y(NAtoms),Z(NAtoms),
                       Alpha(NPrims),CC(NPrims);
   std::vector<int> ShellsPerAtom(NAtoms),Zs(NAtoms),
         PrimsPerShell(NShells),L(NShells);
   for(int i=0;i<NAtoms;i++){
      Zs[i]=PsiMol->Z(i);
      X[i]=PsiMol->x(i);
      Y[i]=PsiMol->y(i);
      Z[i]=PsiMol->z(i);
      ShellsPerAtom[i]=PsiBasis->nshell_on_center(i);
   }
   for(int i=0,total=0;i<NShells;i++){
      PrimsPerShell[i]=PsiBasis->shell(i).nprimitive();
      L[i]=PsiBasis->shell(i).am();
      for(int j=0;j<PrimsPerShell[i];j++){
         Alpha[total]=PsiBasis->shell(i).exp(j);
         CC[total++]=PsiBasis->shell(i).coef(j);
      }
   }
   CInt_importBasisSet((*GTBasis), NAtoms, &Zs[0],
         &X[0], &Y[0], &Z[0], NPrims, NShells, IsPure,
         &ShellsPerAtom[0],&PrimsPerShell[0],&L[0],&CC[0],&Alpha[0]);
}


