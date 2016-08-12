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


#include <libqt/qt.h>
#include <libciomr/libciomr.h>
#include <libmints/mints.h>

#include <libfock/soscf.h>
#include <libdiis/diismanager.h>
#include <libdiis/diisentry.h>

#include "ciwave.h"
#include "civect.h"
#include "structs.h"
#ifdef _HAVEMPI
#include <mpi.h>
#endif

namespace psi { namespace detci {

/*
** compute_mcscf(); Optimizes MO and CI coefficients
**
** Parameters:
**    options = options object
**    alplist = list of alpha strings
**    betlist = list of beta strings
**
** Returns: none
*/
void CIWavefunction::compute_mcscf()
{
  tstart();
  Parameters_->print_lvl = 0;

  boost::shared_ptr<SOMCSCF> somcscf;
  if (MCSCF_Parameters_->mcscf_type == "DF"){
    transform_dfmcscf_ints(!MCSCF_Parameters_->orbital_so);
    somcscf = boost::shared_ptr<SOMCSCF>(new DFSOMCSCF(jk_, dferi_, AO2SO_, H_));
  }
  else if (MCSCF_Parameters_->mcscf_type == "CONV_PARALLEL"){
    transform_mcscf_ints_aodirect(true);
    somcscf = boost::shared_ptr<SOMCSCF>(new IncoreSOMCSCF(jk_, AO2SO_, H_));
  }
  else {
    transform_mcscf_ints(!MCSCF_Parameters_->orbital_so);
    somcscf = boost::shared_ptr<SOMCSCF>(new DiskSOMCSCF(jk_, ints_, AO2SO_, H_));
  }

  somcscf->set_ck_algorithm(options_.get_str("MCSCF_CK_ALGORITHM"));

  // We assume some kind of ras here.
  if (Parameters_->wfn != "CASSCF"){
    std::vector<Dimension> ras_spaces;

    // We only have four spaces currently
    for (int nras = 0; nras < 4; nras++){
      Dimension rasdim = Dimension(nirrep_, "RAS" + psi::to_string(nras));
      for (int h = 0; h < nirrep_; h++){
          rasdim[h] = CalcInfo_->ras_opi[nras][h];
      }
      ras_spaces.push_back(rasdim);
    }
    somcscf->set_ras(ras_spaces);

  }

  // Set fzc energy
  SharedMatrix Cfzc = get_orbitals("FZC");
  somcscf->set_frozen_orbitals(Cfzc);

  /// => Start traditional two-step MCSCF <= //
  // Parameters
  int conv = 0;
  double ediff, grad_rms, current_energy;
  double old_energy = CalcInfo_->escf;
  std::string itertype = "Initial CI";
  std::string mcscf_type;
  if (MCSCF_Parameters_->mcscf_type == "DF"){
    mcscf_type = "   @DF-MCSCF";
    outfile->Printf("\n   ==> Starting DF-MCSCF iterations <==\n\n");
    outfile->Printf("                           "
                      "Total Energy        Delta E      RMS Grad   NCI\n\n");
  }
  else if (MCSCF_Parameters_->mcscf_type == "CONV_PARALLEL"){
    mcscf_type = "   @Parallel-MCSCF";
    outfile->Printf("\n   ==> Starting Parallel-MCSCF iterations <==\n\n");
    outfile->Printf("                           "
                      "Total Energy        Delta E      RMS Grad   NCI  Time(s)\n\n");
  }
  else{
    mcscf_type = "   @MCSCF";
    outfile->Printf("\n   ==> Starting MCSCF iterations <==\n\n");
    outfile->Printf("                        "
                      "Total Energy        Delta E      RMS Grad   NCI  Time(s)\n\n");
  }


  // Setup the DIIS manager
  Dimension dim_oa = get_dimension("DOCC") + get_dimension("ACT");
  Dimension dim_av = get_dimension("VIR") + get_dimension("ACT");

  SharedMatrix x(new Matrix("Rotation Matrix", nirrep_, dim_oa, dim_av));
  SharedMatrix xstep;
  SharedMatrix original_orbs = get_orbitals("ROT");

  boost::shared_ptr<DIISManager> diis_manager(new DIISManager(MCSCF_Parameters_->diis_max_vecs,
                      "MCSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore));
  diis_manager->set_error_vector_size(1, DIISEntry::Matrix, x.get());
  diis_manager->set_vector_size(1, DIISEntry::Matrix, x.get());
  int diis_count = 0;

  bool converged = false;

  // Energy header

  // Iterate
  for (int iter=1; iter<(MCSCF_Parameters_->max_iter + 1); iter++){
    Timer mcscf_iteration;

    int my_rank = 0;
    int nproc   = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Run CI and set quantities
    if(my_rank == 0)
    {
    	Timer solve_ci;
    	diag_h();
    	outfile->Printf("\n Solving CI takes %8.8f s", solve_ci.get());
    	Timer form_pdm;
    	form_opdm();
    	form_tpdm();
    	outfile->Printf("\n Computing PDM took %8.4f s", form_pdm.get());
    }
    MPI_Bcast(&current_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    size_t nact = CalcInfo_->num_ci_orbs
    MPI_Bcast(&(opdm_->pointer()[0]), nact * nact, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    current_energy = Process::environment.globals["MCSCF TOTAL ENERGY"];
    ediff = current_energy - old_energy;

    //// Get orbitals for new update
    SharedMatrix Cdocc = get_orbitals("DOCC");
    SharedMatrix Cact  = get_orbitals("ACT");
    SharedMatrix Cvir  = get_orbitals("VIR");
    SharedMatrix actTPDM = get_tpdm("SUM", true);
    MPI_Bcast(&(actTPDM->pointer()[0]), nact * nact * nact * nact, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (MCSCF_Parameters_->mcscf_type == "CONV_PARALLEL"){
         
        somcscf->set_eri_tensors(tei_aaaa_, tei_raaa_);
    }
    Timer soscf_update;
    somcscf->update(Cdocc, Cact, Cvir, opdm_, actTPDM);
    outfile->Printf("\n MCSCF: Update takes %8.4f s.", soscf_update.get());
    grad_rms = somcscf->gradient_rms();

    if (grad_rms < MCSCF_Parameters_->rms_grad_convergence &&
        (fabs(ediff) < fabs(MCSCF_Parameters_->energy_convergence)) &&
        (iter > 3)){
      outfile->Printf("\n       MCSCF has converged!\n\n");
      converged = true;
      break;
    }
    old_energy = current_energy;

    bool do_so_orbital = (((grad_rms < MCSCF_Parameters_->so_start_grad) &&
                           (ediff < MCSCF_Parameters_->so_start_e) &&
                           (iter >= 2))
                           || (itertype == "SOMCSCF"));

    Timer time_solve;
    if (do_so_orbital && MCSCF_Parameters_->orbital_so){
      itertype = "SOMCSCF";
      xstep = somcscf->solve(3, 1.e-10, false);
    }
    else {
      itertype = "APPROX";
      xstep = somcscf->approx_solve();
    }
    outfile->Printf("\n SOMCSCF Solve takes %8.4f s", time_solve.get());

    Timer scale_x;
    // Scale x if needed
    double maxx = 0.0;
    for (int h=0; h<xstep->nirrep(); ++h) {
        for (int i=0; i<xstep->rowspi()[h]; i++) {
            for (int j=0; j<xstep->colspi()[h]; j++) {
                if (fabs(xstep->get(h,i,j)) > maxx) maxx = fabs(xstep->get(h,i,j));
            }
        }
    }
    outfile->Printf("\n Scale_x takes %8.4f s", scale_x.get());

    if (maxx > MCSCF_Parameters_->max_rot){
      xstep->scale((MCSCF_Parameters_->max_rot)/maxx);
    }

    // Add step to overall rotation
    x->add(xstep);

    // Do we add diis?
    Timer diis_entry;
    if (iter > MCSCF_Parameters_->diis_start){
      diis_manager->add_entry(2, xstep.get(), x.get());
      diis_count++;
    }
    outfile->Printf("\n Adding entry in DIIS takes %8.4f s", diis_entry.get());

    // Do we do diis?
    Timer diis_extra;
    if ((itertype == "APPROX") && !(diis_count % MCSCF_Parameters_->diis_freq) && (iter > MCSCF_Parameters_->diis_start)){
      diis_manager->extrapolate(1, x.get());
      itertype = "APPROX, DIIS";
    }
    outfile->Printf("\n Extrapolatating with DIIS takes %8.4f s", diis_extra.get());

    Timer Ck_time;
    SharedMatrix new_orbs = somcscf->Ck(original_orbs, x);
    outfile->Printf("\n Ck timings take %8.4f s.", Ck_time.get());
    set_orbitals("ROT", new_orbs);

    // Transform integrals
    transform_mcscf_integrals(!MCSCF_Parameters_->orbital_so);
    outfile->Printf("%s Iter %3d:  % 5.14lf   % 1.4e  % 1.4e  %2d   %s   % 1.6e\n", mcscf_type.c_str(), iter,
                    current_energy, ediff, grad_rms, Parameters_->diag_iters_taken,
                    itertype.c_str(), mcscf_iteration.get());


    if(MCSCF_Parameters_->mcscf_type == "CONV_PARALLEL")
    {
        if(get_orbitals("FZC")->ncol() > 0)
        {
            throw PSIEXCEPTION("Frozen core is not implemented for AO_DIRECT, yet");
        }
    }
  }// End MCSCF
  diis_manager->delete_diis_file();
  diis_manager.reset();

  // Do we die if not converged?
  if (!converged){
      outfile->Printf("\nWarning! MCSCF iterations did not fully converge!\n\n");

      // Tricky if it hasnt changed we want to throw
      if (!options_["DIE_IF_NOT_CONVERGED"].has_changed()){
        Parameters_->die_if_not_converged = true;
      }
      convergence_death();
  }


  // Print out the energy
  if (MCSCF_Parameters_->mcscf_type == "DF"){
    outfile->Printf("   @DF-MCSCF Final Energy:  %20.15f\n", current_energy);
  }
  else{
    outfile->Printf("   @MCSCF Final Energy:  %20.15f\n", current_energy);
  }

  // Print root information
  int nroots = Parameters_->num_roots;
  CIvect Dvec(Parameters_->icore, nroots, 1, Parameters_->d_filenum, CIblks_, CalcInfo_,
              Parameters_, H0block_);
  Dvec.init_io_files(true);

  int* mi_iac = init_int_array(Parameters_->nprint);
  int* mi_ibc = init_int_array(Parameters_->nprint);
  int* mi_iaidx = init_int_array(Parameters_->nprint);
  int* mi_ibidx = init_int_array(Parameters_->nprint);
  double* mi_coeff = init_array(Parameters_->nprint);

  outfile->Printf("\n\n   => Energetics <=\n\n");
  outfile->Printf("   SCF energy =             %20.15f\n", CalcInfo_->escf);
  outfile->Printf("   Total CI energy =        %20.15f\n\n", current_energy);

  std::stringstream s;
  outfile->Printf("\n");
  for (int i=0; i<nroots; i++){

   // Print root energy
   s.str(std::string());
   s << "CI ROOT " << (i+1) << " CORRELATION ENERGY";
   outfile->Printf("   CI Root %2d energy =      %20.15f\n", i+1,
       Process::environment.globals[s.str()] + CalcInfo_->escf);

   // Print largest CI coefs
   Dvec.read(i, 0);
   zero_arr(mi_coeff, Parameters_->nprint);
   Dvec.max_abs_vals(Parameters_->nprint, mi_iac, mi_ibc,
      mi_iaidx, mi_ibidx, mi_coeff, Parameters_->neg_only);
   print_vec(Parameters_->nprint, mi_iac, mi_ibc, mi_iaidx, mi_ibidx,
      mi_coeff);
   outfile->Printf("\n");
 }
 print_natural_orbitals();
 free(mi_iac);    free(mi_ibc);
 free(mi_iaidx);  free(mi_ibidx);
 free(mi_coeff);

    tstop();
}

}} // End namespace
