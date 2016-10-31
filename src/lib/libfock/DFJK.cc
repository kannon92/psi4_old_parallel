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
#include <libmints/mints.h>
#include <lib3index/3index.h>
#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libpsio/aiohandler.h>
#include <libqt/qt.h>
#include <psi4-dec.h>
#include <psifiles.h>
#include <libmints/sieve.h>
#include "libmints/local.h"
#include "lib3index/cholesky.h"
#include <libiwl/iwl.hpp>
#include "jk.h"
#include "jk_independent.h"
#include "link.h"
#include "direct_screening.h"
#include "cubature.h"
#include "points.h"

#include<lib3index/cholesky.h>

#include <sstream>
#include "libparallel/ParallelPrinter.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace psi;

namespace psi {

DFJK::DFJK(boost::shared_ptr<BasisSet> primary,
   boost::shared_ptr<BasisSet> auxiliary) :
   JK(primary), auxiliary_(auxiliary)
{
    common_init();
}
DFJK::~DFJK()
{
}
void DFJK::common_init()
{
    df_ints_num_threads_ = 1;
    #ifdef _OPENMP
        df_ints_num_threads_ = omp_get_max_threads();
    #endif
    df_ints_io_ = "NONE";
    condition_ = 1.0E-12;
    unit_ = PSIF_DFSCF_BJ;
    is_core_ = true;
    psio_ = PSIO::shared_object();
}
SharedVector DFJK::iaia(SharedMatrix Ci, SharedMatrix Ca)
{
    // Target quantity
    Dimension dim(Ci->nirrep());
    for (int symm = 0; symm < Ci->nirrep(); symm++) {
        int rank = 0;
        for (int h = 0; h < Ci->nirrep(); h++) {
            rank += Ci->colspi()[h] * Ca->colspi()[h^symm];
        }
        dim[symm] = rank;
    }

    SharedVector Iia(new Vector("(ia|ia)", dim));

    // AO-basis quantities
    int nirrep = Ci->nirrep();
    int nocc = Ci->ncol();
    int nvir = Ca->ncol();
    int nso  = AO2USO_->rowspi()[0];

    SharedMatrix Ci_ao(new Matrix("Ci AO", nso, nocc));
    SharedMatrix Ca_ao(new Matrix("Ca AO", nso, nvir));
    SharedVector Iia_ao(new Vector("(ia|ia) AO", nocc*(ULI)nvir));

    int offset = 0;
    for (int h = 0; h < nirrep; h++) {
        int ni = Ci->colspi()[h];
        int nm = Ci->rowspi()[h];
        if (!ni || !nm) continue;
        double** Cip = Ci->pointer(h);
        double** Cp = Ci_ao->pointer();
        double** Up = AO2USO_->pointer(h);
        C_DGEMM('N','N',nso,ni,nm,1.0,Up[0],nm,Cip[0],ni,0.0,&Cp[0][offset],nocc);
        offset += ni;
    }

    offset = 0;
    for (int h = 0; h < nirrep; h++) {
        int ni = Ca->colspi()[h];
        int nm = Ca->rowspi()[h];
        if (!ni || !nm) continue;
        double** Cip = Ca->pointer(h);
        double** Cp = Ca_ao->pointer();
        double** Up = AO2USO_->pointer(h);
        C_DGEMM('N','N',nso,ni,nm,1.0,Up[0],nm,Cip[0],ni,0.0,&Cp[0][offset],nvir);
        offset += ni;
    }

    // Memory size
    int naux = auxiliary_->nbf();
    int maxrows = max_rows();

    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    const std::vector<long int>& function_pairs_reverse = sieve_->function_pairs_reverse();
    unsigned long int num_nm = function_pairs.size();

    // Temps
    #ifdef _OPENMP
    int temp_nthread = omp_get_max_threads();
    omp_set_num_threads(omp_nthread_);
    C_temp_.resize(omp_nthread_);
    Q_temp_.resize(omp_nthread_);
    #pragma omp parallel
    {
        C_temp_[omp_get_thread_num()] = SharedMatrix(new Matrix("Ctemp", nocc, nso));
        Q_temp_[omp_get_thread_num()] = SharedMatrix(new Matrix("Qtemp", maxrows, nso));
    }
    omp_set_num_threads(temp_nthread);
    #else
    for (int thread = 0; thread < omp_nthread_; thread++) {
        C_temp_.push_back(SharedMatrix(new Matrix("Ctemp", nocc, nso)));
        Q_temp_.push_back(SharedMatrix(new Matrix("Qtemp", maxrows, nso)));
    }
    #endif

    E_left_ = SharedMatrix(new Matrix("E_left", nso, maxrows * nocc));
    E_right_ = SharedMatrix(new Matrix("E_right", nvir, maxrows * nocc));

    // Disk overhead
    psio_address addr = PSIO_ZERO;
    if (!is_core()) {
        Qmn_ = SharedMatrix(new Matrix("(Q|mn) Block", maxrows, num_nm));
        psio_->open(unit_,PSIO_OPEN_OLD);
    }

    // Blocks of Q
    double** Qmnp;
    double** Clp  = Ci_ao->pointer();
    double** Crp  = Ca_ao->pointer();
    double** Elp  = E_left_->pointer();
    double** Erp  = E_right_->pointer();
    double*  Iiap = Iia_ao->pointer();
    for (int Q = 0; Q < naux; Q += maxrows) {

        // Read block of (Q|mn) in
        int rows = (naux - Q <= maxrows ? naux - Q : maxrows);
        if (is_core()) {
            Qmnp = &Qmn_->pointer()[Q];
        } else {
            Qmnp = Qmn_->pointer();
            psio_->read(unit_,"(Q|mn) Integrals", (char*)(Qmn_->pointer()[0]),sizeof(double)*naux*num_nm,addr,&addr);
        }

        // (mi|Q)
        #pragma omp parallel for schedule (dynamic)
        for (int m = 0; m < nso; m++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            double** Ctp = C_temp_[thread]->pointer();
            double** QSp = Q_temp_[thread]->pointer();

            const std::vector<int>& pairs = sieve_->function_to_function()[m];
            int mrows = pairs.size();

            for (int i = 0; i < mrows; i++) {
                int n = pairs[i];
                long int ij = function_pairs_reverse[(m >= n ? (m * (m + 1L) >> 1) + n : (n * (n + 1L) >> 1) + m)];
                C_DCOPY(rows,&Qmnp[0][ij],num_nm,&QSp[0][i],nso);
                C_DCOPY(nocc,Clp[n],1,&Ctp[0][i],nso);
            }

            C_DGEMM('N','T',nocc,rows,mrows,1.0,Ctp[0],nso,QSp[0],nso,0.0,&Elp[0][m*(ULI)nocc*rows],rows);
        }

        // (ai|Q)
        C_DGEMM('T','N',nvir,nocc*(ULI)rows,nso,1.0,Crp[0],nvir,Elp[0],nocc*(ULI)rows,0.0,Erp[0],nocc*(ULI)rows);

        // (ia|Q)(Q|ia)
        for (int i = 0; i < nocc; i++) {
            for (int a = 0; a < nvir; a++) {
                double* Ep = &Erp[0][a * (ULI) nocc * rows + i * rows];
                Iiap[i * nvir + a] += C_DDOT(rows, Ep, 1, Ep, 1);
            }
        }

    }

    // Free disk overhead
    if (!is_core()) {
        Qmn_.reset();
        psio_->close(unit_,1);
    }

    // Free Temps
    E_left_.reset();
    E_right_.reset();
    C_temp_.clear();
    Q_temp_.clear();

    // SO-basis (ia|ia)
    Dimension i_offsets(Ci->nirrep());
    Dimension a_offsets(Ci->nirrep());
    for (int h = 1; h < Ci->nirrep(); h++) {
        i_offsets[h] = i_offsets[h-1] + Ci->colspi()[h-1];
        a_offsets[h] = a_offsets[h-1] + Ca->colspi()[h-1];
    }

    for (int symm = 0; symm < Ci->nirrep(); symm++) {
        offset = 0;
        for (int h = 0; h < Ci->nirrep(); h++) {
            int ni = Ci->colspi()[h];
            int na = Ca->colspi()[h^symm];
            int ioff = i_offsets[h];
            int aoff = a_offsets[h^symm];
            for (int i = 0; i < ni; i++) {
                for (int a = 0; a < na; a++) {
                    Iia->set(symm, i * na + a + offset, Iiap[(ioff + i) * nvir + (aoff + a)]);
                }
            }
            offset += ni * na;
        }
    }

    return Iia;
}
void DFJK::print_header() const
{
    if (print_) {
        outfile->Printf( "  ==> DFJK: Density-Fitted J/K Matrices <==\n\n");

        outfile->Printf( "    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf( "    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf( "    wK tasked:         %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_)
            outfile->Printf( "    Omega:             %11.3E\n", omega_);
        outfile->Printf( "    OpenMP threads:    %11d\n", omp_nthread_);
        outfile->Printf( "    Integrals threads: %11d\n", df_ints_num_threads_);
        outfile->Printf( "    Memory (MB):       %11ld\n", (memory_ *8L) / (1024L * 1024L));
        outfile->Printf( "    Algorithm:         %11s\n",  (is_core_ ? "Core" : "Disk"));
        outfile->Printf( "    Integral Cache:    %11s\n",  df_ints_io_.c_str());
        outfile->Printf( "    Schwarz Cutoff:    %11.0E\n", cutoff_);
        outfile->Printf( "    Fitting Condition: %11.0E\n\n", condition_);

        outfile->Printf( "   => Auxiliary Basis Set <=\n\n");
        auxiliary_->print_by_level("outfile", print_);
    }
}
bool DFJK::is_core() const
{
    size_t ntri = sieve_->function_pairs().size();
    size_t double_size = sizeof(double);
    ULI three_memory = ((ULI)auxiliary_->nbf())*ntri * double_size;;
    ULI two_memory = ((ULI)auxiliary_->nbf())*auxiliary_->nbf() * double_size;

    size_t mem = memory_;
    mem -= memory_overhead();
    mem -= memory_temp();
    // Two is for buffer space in fitting
    if (do_wK_)
        return (3L*three_memory + 2L*two_memory < memory_);
    else
        return ( three_memory + 2L*two_memory < memory_);
}
unsigned long int DFJK::memory_temp() const
{
    unsigned long int mem = 0L;

    // J Overhead (Jtri, Dtri, d)
    mem += 2L * sieve_->function_pairs().size() + auxiliary_->nbf();
    // K Overhead (C_temp, Q_temp)
    mem += omp_nthread_ * (unsigned long int) primary_->nbf() * (auxiliary_->nbf() + max_nocc());

    return mem;
}
int DFJK::max_rows() const
{
    // Start with all memory
    unsigned long int mem = memory_;
    size_t double_size = sizeof(double);
    // Subtract J/K/wK/C/D overhead
    mem -= memory_overhead();
    // Subtract threading temp overhead
    mem -= memory_temp() * double_size;

    // How much will each row cost?
    unsigned long int row_cost = 0L;
    // Copies of E tensor
    row_cost += (lr_symmetric_ ? 1L : 2L) * max_nocc() * primary_->nbf();
    // Slices of Qmn tensor, including AIO buffer (NOTE: AIO not implemented yet)
    row_cost += (is_core_ ? auxiliary_->nbf() : 1L) * sieve_->function_pairs().size();

    unsigned long int max_rows = mem / (double_size * row_cost);

    if (max_rows > (unsigned long int) auxiliary_->nbf())
        max_rows = (unsigned long int) auxiliary_->nbf();
    if (max_rows < 1L)
        max_rows = 1L;

    size_t three_int = max_rows * primary_->nbf() * max_nocc() * 8;
    if(three_int > memory_){
        outfile->Printf("Something is going wrong with memory check: max_rows: %d three_int %lu memory_ %lu", max_rows, three_int, memory_);
        outfile->Printf("\n Auxiliary basis set is %d", auxiliary_->nbf());
        outfile->Printf("\n Setting MAX_ROWS to be 1");
        max_rows = 1L;
    }
    return (int) max_rows;
}
int DFJK::max_nocc() const
{
    int max_nocc = 0;
    for (size_t N = 0; N < C_left_ao_.size(); N++) {
        max_nocc = (C_left_ao_[N]->colspi()[0] > max_nocc ? C_left_ao_[N]->colspi()[0] : max_nocc);
    }
    return max_nocc;
}
void DFJK::initialize_temps()
{
    J_temp_ = boost::shared_ptr<Vector>(new Vector("Jtemp", sieve_->function_pairs().size()));
    D_temp_ = boost::shared_ptr<Vector>(new Vector("Dtemp", sieve_->function_pairs().size()));
    d_temp_ = boost::shared_ptr<Vector>(new Vector("dtemp", max_rows_));


    #ifdef _OPENMP
    int temp_nthread = omp_get_max_threads();
    omp_set_num_threads(omp_nthread_);
    C_temp_.resize(omp_nthread_);
    Q_temp_.resize(omp_nthread_);
    #pragma omp parallel
    {
        C_temp_[omp_get_thread_num()] = SharedMatrix(new Matrix("Ctemp", max_nocc_, primary_->nbf()));
        Q_temp_[omp_get_thread_num()] = SharedMatrix(new Matrix("Qtemp", max_rows_, primary_->nbf()));
    }
    omp_set_num_threads(temp_nthread);
    #else
        for (int thread = 0; thread < omp_nthread_; thread++) {
            C_temp_.push_back(SharedMatrix(new Matrix("Ctemp", max_nocc_, primary_->nbf())));
            Q_temp_.push_back(SharedMatrix(new Matrix("Qtemp", max_rows_, primary_->nbf())));
        }
    #endif

    E_left_ = SharedMatrix(new Matrix("E_left", primary_->nbf(), max_rows_ * max_nocc_));
    if (lr_symmetric_)
        E_right_ = E_left_;
    else
        E_right_ = boost::shared_ptr<Matrix>(new Matrix("E_right", primary_->nbf(), max_rows_ * max_nocc_));

}
void DFJK::initialize_w_temps()
{
    int max_rows_w = max_rows_ / 2;
    max_rows_w = (max_rows_w < 1 ? 1 : max_rows_w);

    #ifdef _OPENMP
    int temp_nthread = omp_get_max_threads();
    omp_set_num_threads(omp_nthread_);
        C_temp_.resize(omp_nthread_);
        Q_temp_.resize(omp_nthread_);
        #pragma omp parallel
        {
            C_temp_[omp_get_thread_num()] = SharedMatrix(new Matrix("Ctemp", max_nocc_, primary_->nbf()));
            Q_temp_[omp_get_thread_num()] = SharedMatrix(new Matrix("Qtemp", max_rows_w, primary_->nbf()));
        }
    omp_set_num_threads(temp_nthread);
    #else
        for (int thread = 0; thread < omp_nthread_; thread++) {
            C_temp_.push_back(SharedMatrix(new Matrix("Ctemp", max_nocc_, primary_->nbf())));
            Q_temp_.push_back(SharedMatrix(new Matrix("Qtemp", max_rows_w, primary_->nbf())));
        }
    #endif

    E_left_  = SharedMatrix(new Matrix("E_left", primary_->nbf(), max_rows_w * max_nocc_));
    E_right_ = SharedMatrix(new Matrix("E_right", primary_->nbf(), max_rows_w * max_nocc_));

}
void DFJK::free_temps()
{
    J_temp_.reset();
    D_temp_.reset();
    d_temp_.reset();
    E_left_.reset();
    E_right_.reset();
    C_temp_.clear();
    Q_temp_.clear();
}
void DFJK::free_w_temps()
{
    E_left_.reset();
    E_right_.reset();
    C_temp_.clear();
    Q_temp_.clear();
}
void DFJK::preiterations()
{

    // DF requires constant sieve, must be static throughout object life
    if (!sieve_) {
        sieve_ = boost::shared_ptr<ERISieve>(new ERISieve(primary_, cutoff_));
    }

    // Core or disk?
    is_core_ =  is_core();
    // KPH is confused.  
    // It seems that is_core() checks whether or not (Q|MN) can fit in core
    // But, max_rows can still be naux..
    // I think that if max_rows() != auxiliary_->nbf() than use disk algorithm
    //if(max_rows() != auxiliary_->nbf())
    //{
    //    is_core_ = false;
    //}


    Timer init_qmn;
    if (is_core_)
        initialize_JK_core();
    else
        initialize_JK_disk();
    outfile->Printf("\n Initialize_DFJK takes %8.4f s.", init_qmn.get());
    outfile->Printf("\n max_nocc: %d max_rows: %d", max_nocc(), max_rows());
    if (do_wK_) {
        if (is_core_)
            initialize_wK_core();
        else
            initialize_wK_disk();
    }
}

void DFJK::compute_JK()
{
    max_nocc_ = max_nocc();
    max_rows_ = max_rows();

    Timer time_dfjk;
    if (do_J_ || do_K_) {
        initialize_temps();
        if (is_core_)
            manage_JK_core();
        else
            manage_JK_disk();
        free_temps();
    }

    if (do_wK_) {
        initialize_w_temps();
        if (is_core_)
            manage_wK_core();
        else
            manage_wK_disk();
        free_w_temps();
        // Bring the wK matrices back to Hermitian
        if (lr_symmetric_) {
            for (size_t N = 0; N < wK_ao_.size(); N++) {
                wK_ao_[N]->hermitivitize();
            }
        }
    }
}
void DFJK::postiterations()
{
    Qmn_.reset();
    Qlmn_.reset();
    Qrmn_.reset();
}
void DFJK::initialize_JK_core()
{
    size_t ntri = sieve_->function_pairs().size();
    ULI three_memory = ((ULI)auxiliary_->nbf())*ntri;
    ULI two_memory = ((ULI)auxiliary_->nbf())*auxiliary_->nbf();

    int nthread = 1;
    #ifdef _OPENMP
        nthread = df_ints_num_threads_;
    #endif
    int rank = 0;

    Qmn_ = SharedMatrix(new Matrix("Qmn (Fitted Integrals)",
        auxiliary_->nbf(), ntri));
    double** Qmnp = Qmn_->pointer();

    // Try to load
    if (df_ints_io_ == "LOAD") {
        psio_->open(unit_,PSIO_OPEN_OLD);
        psio_->read_entry(unit_, "(Q|mn) Integrals", (char*) Qmnp[0], sizeof(double) * ntri * auxiliary_->nbf());
        psio_->close(unit_,1);
        return;
    }

    //Get a TEI for each thread
    boost::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    boost::shared_ptr<IntegralFactory> rifactory(new IntegralFactory(auxiliary_, zero, primary_, primary_));
    const double **buffer = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->eri());
        buffer[Q] = eri[Q]->buffer();
    }

    const std::vector<long int>& schwarz_shell_pairs = sieve_->shell_pairs_reverse();
    const std::vector<long int>& schwarz_fun_pairs = sieve_->function_pairs_reverse();

    int numP,Pshell,MU,NU,P,PHI,mu,nu,nummu,numnu,omu,onu;

    timer_on("JK: (A|mn)");

    //The integrals (A|mn)
    #pragma omp parallel for private (numP, Pshell, MU, NU, P, PHI, mu, nu, nummu, numnu, omu, onu, rank) schedule (dynamic) num_threads(nthread)
    for (MU=0; MU < primary_->nshell(); ++MU) {
        #ifdef _OPENMP
            rank = omp_get_thread_num();
        #endif
        nummu = primary_->shell(MU).nfunction();
        for (NU=0; NU <= MU; ++NU) {
            numnu = primary_->shell(NU).nfunction();
            if (schwarz_shell_pairs[MU*(MU+1)/2+NU] > -1) {
                for (Pshell=0; Pshell < auxiliary_->nshell(); ++Pshell) {
                    numP = auxiliary_->shell(Pshell).nfunction();
                    eri[rank]->compute_shell(Pshell, 0, MU, NU);
                    for (mu=0 ; mu < nummu; ++mu) {
                        omu = primary_->shell(MU).function_index() + mu;
                        for (nu=0; nu < numnu; ++nu) {
                            onu = primary_->shell(NU).function_index() + nu;
                            if(omu>=onu && schwarz_fun_pairs[omu*(omu+1)/2+onu] > -1) {
                                for (P=0; P < numP; ++P) {
                                    PHI = auxiliary_->shell(Pshell).function_index() + P;
                                    Qmnp[PHI][schwarz_fun_pairs[omu*(omu+1)/2+onu]] = buffer[rank][P*nummu*numnu + mu*numnu + nu];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    timer_off("JK: (A|mn)");

    delete []buffer;
    delete []eri;

    timer_on("JK: (A|Q)^-1/2");

    boost::shared_ptr<FittingMetric> Jinv(new FittingMetric(auxiliary_, true));
    Jinv->form_eig_inverse();
    double** Jinvp = Jinv->get_metric()->pointer();

    timer_off("JK: (A|Q)^-1/2");

    ULI max_cols = (memory_-three_memory-two_memory) / auxiliary_->nbf();
    if (max_cols < 1)
        max_cols = 1;
    if (max_cols > ntri)
        max_cols = ntri;
    SharedMatrix temp(new Matrix("Qmn buffer", auxiliary_->nbf(), max_cols));
    double** tempp = temp->pointer();

    size_t nblocks = ntri / max_cols;
    if ((ULI)nblocks*max_cols != ntri) nblocks++;

    size_t ncol = 0;
    size_t col = 0;

    timer_on("JK: (Q|mn)");

    for (size_t block = 0; block < nblocks; block++) {

        ncol = max_cols;
        if (col + ncol > ntri)
            ncol = ntri - col;

        C_DGEMM('N','N',auxiliary_->nbf(), ncol, auxiliary_->nbf(), 1.0,
            Jinvp[0], auxiliary_->nbf(), &Qmnp[0][col], ntri, 0.0,
            tempp[0], max_cols);

        for (int Q = 0; Q < auxiliary_->nbf(); Q++) {
            C_DCOPY(ncol, tempp[Q], 1, &Qmnp[Q][col], 1);
        }

        col += ncol;
    }

    timer_off("JK: (Q|mn)");

    if (df_ints_io_ == "SAVE") {
        psio_->open(unit_,PSIO_OPEN_NEW);
        psio_->write_entry(unit_, "(Q|mn) Integrals", (char*) Qmnp[0], sizeof(double) * ntri * auxiliary_->nbf());
        psio_->close(unit_,1);
    }
}
void DFJK::initialize_JK_disk()
{
    // Try to load
    if (df_ints_io_ == "LOAD") {
        return;
    }

    int nshell = primary_->nshell();
    int naux = auxiliary_->nbf();

    // ==> Schwarz Indexing <== //
    const std::vector<std::pair<int,int> >& schwarz_shell_pairs = sieve_->shell_pairs();
    const std::vector<std::pair<int,int> >& schwarz_fun_pairs = sieve_->function_pairs();
    int nshellpairs = schwarz_shell_pairs.size();
    int ntri = schwarz_fun_pairs.size();
    const std::vector<long int>&  schwarz_shell_pairs_r = sieve_->shell_pairs_reverse();
    const std::vector<long int>&  schwarz_fun_pairs_r = sieve_->function_pairs_reverse();

    // ==> Memory Sizing <== //
    ULI two_memory = ((ULI)auxiliary_->nbf())*auxiliary_->nbf();
    ULI buffer_memory = memory_ - 2*two_memory; // Two is for buffer space in fitting

    //outfile->Printf( "Buffer memory = %ld words\n", buffer_memory);

    //outfile->Printf("Schwarz Shell Pairs:\n");
    //for (int MN = 0; MN < nshellpairs; MN++) {
    //    outfile->Printf("  %3d: (%3d,%3d)\n", MN, schwarz_shell_pairs[2*MN], schwarz_shell_pairs[2*MN + 1]);
    //}

    //outfile->Printf("Schwarz Function Pairs:\n");
    //for (int MN = 0; MN < ntri; MN++) {
    //    outfile->Printf("  %3d: (%3d,%3d)\n", MN, schwarz_fun_pairs[2*MN], schwarz_fun_pairs[2*MN + 1]);
    //}

    //outfile->Printf("Schwarz Reverse Shell Pairs:\n");
    //for (int MN = 0; MN < primary_->nshell() * (primary_->nshell() + 1) / 2; MN++) {
    //    outfile->Printf("  %3d: %4ld\n", MN, schwarz_shell_pairs_r[MN]);
    //}

    //outfile->Printf("Schwarz Reverse Function Pairs:\n");
    //for (int MN = 0; MN < primary_->nbf() * (primary_->nbf() + 1) / 2; MN++) {
    //    outfile->Printf("  %3d: %4ld\n", MN, schwarz_fun_pairs_r[MN]);
    //}

    // Find out exactly how much memory per MN shell
    boost::shared_ptr<IntVector> MN_mem(new IntVector("Memory per MN pair", nshell * (nshell + 1) / 2));
    int *MN_memp = MN_mem->pointer();

    for (int mn = 0; mn < ntri; mn++) {
        int m = schwarz_fun_pairs[mn].first;
        int n = schwarz_fun_pairs[mn].second;

        int M = primary_->function_to_shell(m);
        int N = primary_->function_to_shell(n);

        MN_memp[M * (M + 1) / 2 + N] += naux;
    }

    //MN_mem->print(outfile);

    // Figure out exactly how much memory per M row
    ULI* M_memp = new ULI[nshell];
    memset(static_cast<void*>(M_memp), '\0', nshell*sizeof(ULI));

    for (int M = 0; M < nshell; M++) {
        for (int N = 0; N <= M; N++) {
            M_memp[M] += MN_memp[M * (M + 1) / 2 + N];
        }
    }

    //outfile->Printf("  # Memory per M row #\n\n");
    //for (int M = 0; M < nshell; M++)
    //    outfile->Printf("   %3d: %10ld\n", M+1,M_memp[M]);
    //outfile->Printf("\n");

    // Find and check the minimum required memory for this problem
    ULI min_mem = naux*(ULI) ntri;
    for (int M = 0; M < nshell; M++) {
        if (min_mem > M_memp[M])
            min_mem = M_memp[M];
    }

    if (min_mem > buffer_memory) {
        std::stringstream message;
        message << "SCF::DF: Disk based algorithm requires 2 (A|B) fitting metrics and an (A|mn) chunk on core." << std::endl;
        message << "         This is 2Q^2 + QNP doubles, where Q is the auxiliary basis size, N is the" << std::endl;
        message << "         primary basis size, and P is the maximum number of functions in a primary shell." << std::endl;
        message << "         For this problem, that is " << ((8L*(min_mem + 2*two_memory))) << " bytes before taxes,";
        message << ((80L*(min_mem + 2*two_memory) / 7L)) << " bytes after taxes. " << std::endl;

        throw PSIEXCEPTION(message.str());
    }

    // ==> Reduced indexing by M <== //

    // Figure out the MN start index per M row
    boost::shared_ptr<IntVector> MN_start(new IntVector("MUNU start per M row", nshell));
    int* MN_startp = MN_start->pointer();

    MN_startp[0] = schwarz_shell_pairs_r[0];
    int M_index = 1;
    for (int MN = 0; MN < nshellpairs; MN++) {
        if (schwarz_shell_pairs[MN].first == M_index) {
            MN_startp[M_index] = MN;
            M_index++;
        }
    }

    // Figure out the mn start index per M row
    boost::shared_ptr<IntVector> mn_start(new IntVector("munu start per M row", nshell));
    int* mn_startp = mn_start->pointer();

    mn_startp[0] = schwarz_fun_pairs[0].first;
    int m_index = 1;
    for (int mn = 0; mn < ntri; mn++) {
        if (primary_->function_to_shell(schwarz_fun_pairs[mn].first) == m_index) {
            mn_startp[m_index] = mn;
            m_index++;
        }
    }

    // Figure out the MN columns per M row
    boost::shared_ptr<IntVector> MN_col(new IntVector("MUNU cols per M row", nshell));
    int* MN_colp = MN_col->pointer();

    for (int M = 1; M < nshell; M++) {
        MN_colp[M - 1] = MN_startp[M] - MN_startp[M - 1];
    }
    MN_colp[nshell - 1] = nshellpairs - MN_startp[nshell - 1];

    // Figure out the mn columns per M row
    boost::shared_ptr<IntVector> mn_col(new IntVector("munu cols per M row", nshell));
    int* mn_colp = mn_col->pointer();

    for (int M = 1; M < nshell; M++) {
        mn_colp[M - 1] = mn_startp[M] - mn_startp[M - 1];
    }
    mn_colp[nshell - 1] = ntri - mn_startp[nshell - 1];

    //MN_start->print(outfile);
    //MN_col->print(outfile);
    //mn_start->print(outfile);
    //mn_col->print(outfile);

    // ==> Block indexing <== //
    // Sizing by block
    std::vector<int> MN_start_b;
    std::vector<int> MN_col_b;
    std::vector<int> mn_start_b;
    std::vector<int> mn_col_b;

    // Determine MN and mn block starts
    // also MN and mn block cols
    int nblock = 1;
    ULI current_mem = 0L;
    MN_start_b.push_back(0);
    mn_start_b.push_back(0);
    MN_col_b.push_back(0);
    mn_col_b.push_back(0);
    for (int M = 0; M < nshell; M++) {
        if (current_mem + M_memp[M] > buffer_memory) {
            MN_start_b.push_back(MN_startp[M]);
            mn_start_b.push_back(mn_startp[M]);
            MN_col_b.push_back(0);
            mn_col_b.push_back(0);
            nblock++;
            current_mem = 0L;
        }
        MN_col_b[nblock - 1] += MN_colp[M];
        mn_col_b[nblock - 1] += mn_colp[M];
        current_mem += M_memp[M];
    }

    //outfile->Printf("Block, MN start, MN cols, mn start, mn cols\n");
    //for (int block = 0; block < nblock; block++) {
    //    outfile->Printf("  %3d: %12d %12d %12d %12d\n", block, MN_start_b[block], MN_col_b[block], mn_start_b[block], mn_col_b[block]);
    //}
    //

    // Full sizing not required any longer
    MN_mem.reset();
    MN_start.reset();
    MN_col.reset();
    mn_start.reset();
    mn_col.reset();
    delete[] M_memp;

    // ==> Buffer allocation <== //
    int max_cols = 0;
    for (int block = 0; block < nblock; block++) {
        if (max_cols < mn_col_b[block])
            max_cols = mn_col_b[block];
    }

    // Primary buffer
    Qmn_ = SharedMatrix(new Matrix("(Q|mn) (Disk Chunk)", naux, max_cols));
    // Fitting buffer
    SharedMatrix Amn (new Matrix("(Q|mn) (Buffer)",naux,naux));
    double** Qmnp = Qmn_->pointer();
    double** Amnp = Amn->pointer();

    // ==> Prestripe/Jinv <== //

    timer_on("JK: (A|Q)^-1");

    psio_->open(unit_,PSIO_OPEN_NEW);
    boost::shared_ptr<AIOHandler> aio(new AIOHandler(psio_));

    // Dispatch the prestripe
    aio->zero_disk(unit_,"(Q|mn) Integrals",naux,ntri);

    // Form the J symmetric inverse
    boost::shared_ptr<FittingMetric> Jinv(new FittingMetric(auxiliary_, true));
    Jinv->form_eig_inverse();
    double** Jinvp = Jinv->get_metric()->pointer();

    // Synch up
    aio->synchronize();

    timer_off("JK: (A|Q)^-1");

    // ==> Thread setup <== //
    int nthread = 1;
    #ifdef _OPENMP
        nthread = df_ints_num_threads_;
    #endif

    // ==> ERI initialization <== //
    boost::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    boost::shared_ptr<IntegralFactory> rifactory(new IntegralFactory(auxiliary_, zero, primary_, primary_));
    const double **buffer = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->eri());
        buffer[Q] = eri[Q]->buffer();
    }

    // ==> Main loop <== //
    for (int block = 0; block < nblock; block++) {
        int MN_start_val = MN_start_b[block];
        int mn_start_val = mn_start_b[block];
        int MN_col_val = MN_col_b[block];
        int mn_col_val = mn_col_b[block];

        // ==> (A|mn) integrals <== //

        timer_on("JK: (A|mn)");

        #pragma omp parallel for schedule(guided) num_threads(nthread)
        for (int MUNU = MN_start_val; MUNU < MN_start_val + MN_col_val; MUNU++) {

            int rank = 0;
            #ifdef _OPENMP
                rank = omp_get_thread_num();
            #endif

            int MU = schwarz_shell_pairs[MUNU + 0].first;
            int NU = schwarz_shell_pairs[MUNU + 0].second;
            int nummu = primary_->shell(MU).nfunction();
            int numnu = primary_->shell(NU).nfunction();
            int mu = primary_->shell(MU).function_index();
            int nu = primary_->shell(NU).function_index();
            for (int P = 0; P < auxiliary_->nshell(); P++) {
                int nump = auxiliary_->shell(P).nfunction();
                int p = auxiliary_->shell(P).function_index();
                eri[rank]->compute_shell(P,0,MU,NU);
                for (int dm = 0; dm < nummu; dm++) {
                    int omu = mu + dm;
                    for (int dn = 0; dn < numnu;  dn++) {
                        int onu = nu + dn;
                        if (omu >= onu && schwarz_fun_pairs_r[omu*(omu+1)/2 + onu] >= 0) {
                            int delta = schwarz_fun_pairs_r[omu*(omu+1)/2 + onu] - mn_start_val;
                            for (int dp = 0; dp < nump; dp ++) {
                                int op = p + dp;
                                Qmnp[op][delta] = buffer[rank][dp*nummu*numnu + dm*numnu + dn];
                            }
                        }
                    }
                }
            }
        }

        timer_off("JK: (A|mn)");

        // ==> (Q|mn) fitting <== //

        timer_on("JK: (Q|mn)");

        for (int mn = 0; mn < mn_col_val; mn+=naux) {
            int cols = naux;
            if (mn + naux >= mn_col_val)
                cols = mn_col_val - mn;

            for (int Q = 0; Q < naux; Q++)
                C_DCOPY(cols,&Qmnp[Q][mn],1,Amnp[Q],1);

            C_DGEMM('N','N',naux,cols,naux,1.0,Jinvp[0],naux,Amnp[0],naux,0.0,&Qmnp[0][mn],max_cols);
        }

        timer_off("JK: (Q|mn)");

        // ==> Disk striping <== //

        timer_on("JK: (Q|mn) Write");

        psio_address addr;
        for (int Q = 0; Q < naux; Q++) {
            addr = psio_get_address(PSIO_ZERO, (Q*(ULI) ntri + mn_start_val)*sizeof(double));
            psio_->write(unit_,"(Q|mn) Integrals", (char*)Qmnp[Q],mn_col_val*sizeof(double),addr,&addr);
        }

        timer_off("JK: (Q|mn) Write");
    }

    // ==> Close out <== //
    Qmn_.reset();
    delete[] eri;

    psio_->close(unit_,1);
}
void DFJK::initialize_wK_core()
{
    int naux = auxiliary_->nbf();
    int ntri = sieve_->function_pairs().size();

    int nthread = 1;
    #ifdef _OPENMP
        nthread = df_ints_num_threads_;
    #endif
    int rank = 0;

    // Check that the right integrals are using the correct omega
    if (df_ints_io_ == "LOAD") {
        psio_->open(unit_,PSIO_OPEN_OLD);
        double check_omega;
        psio_->read_entry(unit_, "Omega", (char*)&check_omega, sizeof(double));
        if (check_omega != omega_) {
            rebuild_wK_disk();
        }
        psio_->close(unit_,1);
    }

    Qlmn_ = SharedMatrix(new Matrix("Qlmn (Fitted Integrals)",
        auxiliary_->nbf(), ntri));
    double** Qmnp = Qlmn_->pointer();

    Qrmn_ = SharedMatrix(new Matrix("Qrmn (Fitted Integrals)",
        auxiliary_->nbf(), ntri));
    double** Qmn2p = Qrmn_->pointer();

    // Try to load
    if (df_ints_io_ == "LOAD") {
        psio_->open(unit_,PSIO_OPEN_OLD);
        psio_->read_entry(unit_, "Left (Q|w|mn) Integrals", (char*) Qmnp[0], sizeof(double) * ntri * auxiliary_->nbf());
        psio_->read_entry(unit_, "Right (Q|w|mn) Integrals", (char*) Qmn2p[0], sizeof(double) * ntri * auxiliary_->nbf());
        psio_->close(unit_,1);
        return;
    }

    // => Left Integrals <= //

    //Get a TEI for each thread
    boost::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    boost::shared_ptr<IntegralFactory> rifactory(new IntegralFactory(auxiliary_, zero, primary_, primary_));
    const double **buffer = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->eri());
        buffer[Q] = eri[Q]->buffer();
    }

    const std::vector<long int>& schwarz_shell_pairs = sieve_->shell_pairs_reverse();
    const std::vector<long int>& schwarz_fun_pairs = sieve_->function_pairs_reverse();

    int numP,Pshell,MU,NU,P,PHI,mu,nu,nummu,numnu,omu,onu;
    //The integrals (A|mn)

    timer_on("JK: (A|mn)^L");

    #pragma omp parallel for private (numP, Pshell, MU, NU, P, PHI, mu, nu, nummu, numnu, omu, onu, rank) schedule (dynamic) num_threads(nthread)
    for (MU=0; MU < primary_->nshell(); ++MU) {
        #ifdef _OPENMP
            rank = omp_get_thread_num();
        #endif
        nummu = primary_->shell(MU).nfunction();
        for (NU=0; NU <= MU; ++NU) {
            numnu = primary_->shell(NU).nfunction();
            if (schwarz_shell_pairs[MU*(MU+1)/2+NU] > -1) {
                for (Pshell=0; Pshell < auxiliary_->nshell(); ++Pshell) {
                    numP = auxiliary_->shell(Pshell).nfunction();
                    eri[rank]->compute_shell(Pshell, 0, MU, NU);
                    for (mu=0 ; mu < nummu; ++mu) {
                        omu = primary_->shell(MU).function_index() + mu;
                        for (nu=0; nu < numnu; ++nu) {
                            onu = primary_->shell(NU).function_index() + nu;
                            if(omu>=onu && schwarz_fun_pairs[omu*(omu+1)/2+onu] > -1) {
                                for (P=0; P < numP; ++P) {
                                    PHI = auxiliary_->shell(Pshell).function_index() + P;
                                    Qmn2p[PHI][schwarz_fun_pairs[omu*(omu+1)/2+onu]] = buffer[rank][P*nummu*numnu + mu*numnu + nu];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    timer_off("JK: (A|mn)^L");

    delete []buffer;
    delete []eri;

    // => Fitting <= //

    timer_on("JK: (A|Q)^-1");

    // Fitting metric
    boost::shared_ptr<FittingMetric> Jinv(new FittingMetric(auxiliary_, true));
    Jinv->form_full_eig_inverse();
    double** Jinvp = Jinv->get_metric()->pointer();

    timer_off("JK: (A|Q)^-1");

    timer_on("JK: (Q|mn)^L");

    // Fitting in one GEMM (being a clever bastard)
    C_DGEMM('N','N',naux,ntri,naux,1.0,Jinvp[0],naux,Qmn2p[0],ntri,0.0,Qmnp[0],ntri);

    timer_off("JK: (Q|mn)^L");

    // => Right Integrals <= //

    const double **buffer2 = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri2 = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri2[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->erf_eri(omega_));
        buffer2[Q] = eri2[Q]->buffer();
    }

    //The integrals (A|w|mn)

    timer_on("JK: (A|mn)^R");

    #pragma omp parallel for private (numP, Pshell, MU, NU, P, PHI, mu, nu, nummu, numnu, omu, onu, rank) schedule (dynamic) num_threads(nthread)
    for (MU=0; MU < primary_->nshell(); ++MU) {
        #ifdef _OPENMP
            rank = omp_get_thread_num();
        #endif
        nummu = primary_->shell(MU).nfunction();
        for (NU=0; NU <= MU; ++NU) {
            numnu = primary_->shell(NU).nfunction();
            if (schwarz_shell_pairs[MU*(MU+1)/2+NU] > -1) {
                for (Pshell=0; Pshell < auxiliary_->nshell(); ++Pshell) {
                    numP = auxiliary_->shell(Pshell).nfunction();
                    eri2[rank]->compute_shell(Pshell, 0, MU, NU);
                    for (mu=0 ; mu < nummu; ++mu) {
                        omu = primary_->shell(MU).function_index() + mu;
                        for (nu=0; nu < numnu; ++nu) {
                            onu = primary_->shell(NU).function_index() + nu;
                            if(omu>=onu && schwarz_fun_pairs[omu*(omu+1)/2+onu] > -1) {
                                for (P=0; P < numP; ++P) {
                                    PHI = auxiliary_->shell(Pshell).function_index() + P;
                                    Qmn2p[PHI][schwarz_fun_pairs[omu*(omu+1)/2+onu]] = buffer2[rank][P*nummu*numnu + mu*numnu + nu];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    timer_off("JK: (A|mn)^R");

    delete []buffer2;
    delete []eri2;

    // Try to save
    if (df_ints_io_ == "SAVE") {
        psio_->open(unit_,PSIO_OPEN_OLD);
        psio_->write_entry(unit_, "Left (Q|w|mn) Integrals", (char*) Qmnp[0], sizeof(double) * ntri * auxiliary_->nbf());
        psio_->write_entry(unit_, "Right (Q|w|mn) Integrals", (char*) Qmn2p[0], sizeof(double) * ntri * auxiliary_->nbf());
        psio_->write_entry(unit_, "Omega", (char*) &omega_, sizeof(double));
        psio_->close(unit_,1);
    }
}
void DFJK::initialize_wK_disk()
{
    // Try to load
    if (df_ints_io_ == "LOAD") {
        psio_->open(unit_,PSIO_OPEN_OLD);
        double check_omega;
        psio_->read_entry(unit_, "Omega", (char*)&check_omega, sizeof(double));
        if (check_omega != omega_) {
            rebuild_wK_disk();
        }
        psio_->close(unit_,1);
    }

    size_t nshell = primary_->nshell();
    size_t naux = auxiliary_->nbf();

    // ==> Schwarz Indexing <== //
    const std::vector<std::pair<int,int> >& schwarz_shell_pairs = sieve_->shell_pairs();
    const std::vector<std::pair<int,int> >& schwarz_fun_pairs = sieve_->function_pairs();
    int nshellpairs = schwarz_shell_pairs.size();
    int ntri = schwarz_fun_pairs.size();
    const std::vector<long int>&  schwarz_shell_pairs_r = sieve_->shell_pairs_reverse();
    const std::vector<long int>&  schwarz_fun_pairs_r = sieve_->function_pairs_reverse();

    // ==> Memory Sizing <== //
    ULI two_memory = ((ULI)auxiliary_->nbf())*auxiliary_->nbf();
    ULI buffer_memory = memory_ - 2L*two_memory; // Two is for buffer space in fitting

    //outfile->Printf( "Buffer memory = %ld words\n", buffer_memory);

    //outfile->Printf("Schwarz Shell Pairs:\n");
    //for (int MN = 0; MN < nshellpairs; MN++) {
    //    outfile->Printf("  %3d: (%3d,%3d)\n", MN, schwarz_shell_pairs[2*MN], schwarz_shell_pairs[2*MN + 1]);
    //}

    //outfile->Printf("Schwarz Function Pairs:\n");
    //for (int MN = 0; MN < ntri; MN++) {
    //    outfile->Printf("  %3d: (%3d,%3d)\n", MN, schwarz_fun_pairs[2*MN], schwarz_fun_pairs[2*MN + 1]);
    //}

    //outfile->Printf("Schwarz Reverse Shell Pairs:\n");
    //for (int MN = 0; MN < primary_->nshell() * (primary_->nshell() + 1) / 2; MN++) {
    //    outfile->Printf("  %3d: %4ld\n", MN, schwarz_shell_pairs_r[MN]);
    //}

    //outfile->Printf("Schwarz Reverse Function Pairs:\n");
    //for (int MN = 0; MN < primary_->nbf() * (primary_->nbf() + 1) / 2; MN++) {
    //    outfile->Printf("  %3d: %4ld\n", MN, schwarz_fun_pairs_r[MN]);
    //}

    // Find out exactly how much memory per MN shell
    boost::shared_ptr<IntVector> MN_mem(new IntVector("Memory per MN pair", nshell * (nshell + 1) / 2));
    int *MN_memp = MN_mem->pointer();

    for (int mn = 0; mn < ntri; mn++) {
        int m = schwarz_fun_pairs[mn].first;
        int n = schwarz_fun_pairs[mn].second;

        int M = primary_->function_to_shell(m);
        int N = primary_->function_to_shell(n);

        MN_memp[M * (M + 1) / 2 + N] += naux;
    }

    //MN_mem->print(outfile);

    // Figure out exactly how much memory per M row
    ULI* M_memp = new ULI[nshell];
    memset(static_cast<void*>(M_memp), '\0', nshell*sizeof(ULI));

    for (size_t M = 0; M < nshell; M++) {
        for (size_t N = 0; N <= M; N++) {
            M_memp[M] += MN_memp[M * (M + 1) / 2 + N];
        }
    }

    //outfile->Printf("  # Memory per M row #\n\n");
    //for (int M = 0; M < nshell; M++)
    //    outfile->Printf("   %3d: %10ld\n", M+1,M_memp[M]);
    //outfile->Printf("\n");

    // Find and check the minimum required memory for this problem
    ULI min_mem = naux*(ULI) ntri;
    for (size_t M = 0; M < nshell; M++) {
        if (min_mem > M_memp[M])
            min_mem = M_memp[M];
    }

    if (min_mem > buffer_memory) {
        std::stringstream message;
        message << "SCF::DF: Disk based algorithm requires 2 (A|B) fitting metrics and an (A|mn) chunk on core." << std::endl;
        message << "         This is 2Q^2 + QNP doubles, where Q is the auxiliary basis size, N is the" << std::endl;
        message << "         primary basis size, and P is the maximum number of functions in a primary shell." << std::endl;
        message << "         For this problem, that is " << ((8L*(min_mem + 2*two_memory))) << " bytes before taxes,";
        message << ((80L*(min_mem + 2*two_memory) / 7L)) << " bytes after taxes. " << std::endl;

        throw PSIEXCEPTION(message.str());
    }

    // ==> Reduced indexing by M <== //

    // Figure out the MN start index per M row
    boost::shared_ptr<IntVector> MN_start(new IntVector("MUNU start per M row", nshell));
    int* MN_startp = MN_start->pointer();

    MN_startp[0] = schwarz_shell_pairs_r[0];
    int M_index = 1;
    for (int MN = 0; MN < nshellpairs; MN++) {
        if (schwarz_shell_pairs[MN].first == M_index) {
            MN_startp[M_index] = MN;
            M_index++;
        }
    }

    // Figure out the mn start index per M row
    boost::shared_ptr<IntVector> mn_start(new IntVector("munu start per M row", nshell));
    int* mn_startp = mn_start->pointer();

    mn_startp[0] = schwarz_fun_pairs[0].first;
    int m_index = 1;
    for (int mn = 0; mn < ntri; mn++) {
        if (primary_->function_to_shell(schwarz_fun_pairs[mn].first) == m_index) {
            mn_startp[m_index] = mn;
            m_index++;
        }
    }

    // Figure out the MN columns per M row
    boost::shared_ptr<IntVector> MN_col(new IntVector("MUNU cols per M row", nshell));
    int* MN_colp = MN_col->pointer();

    for (size_t M = 1; M < nshell; M++) {
        MN_colp[M - 1] = MN_startp[M] - MN_startp[M - 1];
    }
    MN_colp[nshell - 1] = nshellpairs - MN_startp[nshell - 1];

    // Figure out the mn columns per M row
    boost::shared_ptr<IntVector> mn_col(new IntVector("munu cols per M row", nshell));
    int* mn_colp = mn_col->pointer();

    for (size_t M = 1; M < nshell; M++) {
        mn_colp[M - 1] = mn_startp[M] - mn_startp[M - 1];
    }
    mn_colp[nshell - 1] = ntri - mn_startp[nshell - 1];

    //MN_start->print(outfile);
    //MN_col->print(outfile);
    //mn_start->print(outfile);
    //mn_col->print(outfile);

    // ==> Block indexing <== //
    // Sizing by block
    std::vector<int> MN_start_b;
    std::vector<int> MN_col_b;
    std::vector<int> mn_start_b;
    std::vector<int> mn_col_b;

    // Determine MN and mn block starts
    // also MN and mn block cols
    int nblock = 1;
    ULI current_mem = 0L;
    MN_start_b.push_back(0);
    mn_start_b.push_back(0);
    MN_col_b.push_back(0);
    mn_col_b.push_back(0);
    for (size_t M = 0; M < nshell; M++) {
        if (current_mem + M_memp[M] > buffer_memory) {
            MN_start_b.push_back(MN_startp[M]);
            mn_start_b.push_back(mn_startp[M]);
            MN_col_b.push_back(0);
            mn_col_b.push_back(0);
            nblock++;
            current_mem = 0L;
        }
        MN_col_b[nblock - 1] += MN_colp[M];
        mn_col_b[nblock - 1] += mn_colp[M];
        current_mem += M_memp[M];
    }

    //outfile->Printf("Block, MN start, MN cols, mn start, mn cols\n");
    //for (int block = 0; block < nblock; block++) {
    //    outfile->Printf("  %3d: %12d %12d %12d %12d\n", block, MN_start_b[block], MN_col_b[block], mn_start_b[block], mn_col_b[block]);
    //}
    //

    // Full sizing not required any longer
    MN_mem.reset();
    MN_start.reset();
    MN_col.reset();
    mn_start.reset();
    mn_col.reset();
    delete[] M_memp;

    // ==> Buffer allocation <== //
    int max_cols = 0;
    for (int block = 0; block < nblock; block++) {
        if (max_cols < mn_col_b[block])
            max_cols = mn_col_b[block];
    }

    // Primary buffer
    Qmn_ = SharedMatrix(new Matrix("(Q|mn) (Disk Chunk)", naux, max_cols));
    // Fitting buffer
    SharedMatrix Amn (new Matrix("(Q|mn) (Buffer)",naux,naux));
    double** Qmnp = Qmn_->pointer();
    double** Amnp = Amn->pointer();

    // ==> Prestripe/Jinv <== //
    psio_->open(unit_,PSIO_OPEN_OLD);
    boost::shared_ptr<AIOHandler> aio(new AIOHandler(psio_));

    // Dispatch the prestripe
    aio->zero_disk(unit_,"Left (Q|w|mn) Integrals",naux,ntri);

    // Form the J full inverse
    boost::shared_ptr<FittingMetric> Jinv(new FittingMetric(auxiliary_, true));
    Jinv->form_full_eig_inverse();
    double** Jinvp = Jinv->get_metric()->pointer();

    // Synch up
    aio->synchronize();

    // ==> Thread setup <== //
    int nthread = 1;
    #ifdef _OPENMP
        nthread = df_ints_num_threads_;
    #endif

    // ==> ERI initialization <== //
    boost::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    boost::shared_ptr<IntegralFactory> rifactory(new IntegralFactory(auxiliary_, zero, primary_, primary_));
    const double **buffer = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->eri());
        buffer[Q] = eri[Q]->buffer();
    }

    // ==> Main loop <== //
    for (int block = 0; block < nblock; block++) {
        int MN_start_val = MN_start_b[block];
        int mn_start_val = mn_start_b[block];
        int MN_col_val = MN_col_b[block];
        int mn_col_val = mn_col_b[block];

        // ==> (A|mn) integrals <== //

        timer_on("JK: (A|mn)^L");

        #pragma omp parallel for schedule(guided) num_threads(nthread)
        for (int MUNU = MN_start_val; MUNU < MN_start_val + MN_col_val; MUNU++) {

            int rank = 0;
            #ifdef _OPENMP
                rank = omp_get_thread_num();
            #endif

            int MU = schwarz_shell_pairs[MUNU + 0].first;
            int NU = schwarz_shell_pairs[MUNU + 0].second;
            int nummu = primary_->shell(MU).nfunction();
            int numnu = primary_->shell(NU).nfunction();
            int mu = primary_->shell(MU).function_index();
            int nu = primary_->shell(NU).function_index();
            for (int P = 0; P < auxiliary_->nshell(); P++) {
                int nump = auxiliary_->shell(P).nfunction();
                int p = auxiliary_->shell(P).function_index();
                eri[rank]->compute_shell(P,0,MU,NU);
                for (int dm = 0; dm < nummu; dm++) {
                    int omu = mu + dm;
                    for (int dn = 0; dn < numnu;  dn++) {
                        int onu = nu + dn;
                        if (omu >= onu && schwarz_fun_pairs_r[omu*(omu+1)/2 + onu] >= 0) {
                            int delta = schwarz_fun_pairs_r[omu*(omu+1)/2 + onu] - mn_start_val;
                            for (int dp = 0; dp < nump; dp ++) {
                                int op = p + dp;
                                Qmnp[op][delta] = buffer[rank][dp*nummu*numnu + dm*numnu + dn];
                            }
                        }
                    }
                }
            }
        }

        timer_off("JK: (A|mn)^L");

        // ==> (Q|mn) fitting <== //

        timer_on("JK: (Q|mn)^L");

        for (int mn = 0; mn < mn_col_val; mn+=naux) {
            int cols = naux;
            if (mn + naux >= (size_t)mn_col_val)
                cols = mn_col_val - mn;

            for (size_t Q = 0; Q < naux; Q++)
                C_DCOPY(cols,&Qmnp[Q][mn],1,Amnp[Q],1);

            C_DGEMM('N','N',naux,cols,naux,1.0,Jinvp[0],naux,Amnp[0],naux,0.0,&Qmnp[0][mn],max_cols);
        }

        timer_off("JK: (Q|mn)^L");

        // ==> Disk striping <== //

        timer_on("JK: (Q|mn)^L Write");

        psio_address addr;
        for (size_t Q = 0; Q < naux; Q++) {
            addr = psio_get_address(PSIO_ZERO, (Q*(ULI) ntri + mn_start_val)*sizeof(double));
            psio_->write(unit_,"Left (Q|w|mn) Integrals", (char*)Qmnp[Q],mn_col_val*sizeof(double),addr,&addr);
        }

        timer_off("JK: (Q|mn)^L Write");

    }

    Qmn_.reset();
    delete[] eri;

    // => Right Integrals <= //

    const double **buffer2 = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri2 = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri2[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->erf_eri(omega_));
        buffer2[Q] = eri2[Q]->buffer();
    }

    ULI maxP = auxiliary_->max_function_per_shell();
    ULI max_rows = memory_ / ntri;
    max_rows = (max_rows > naux ? naux : max_rows);
    max_rows = (max_rows < maxP ? maxP : max_rows);

    // Block extents
    std::vector<int> block_Q_starts;
    size_t counter = 0;
    block_Q_starts.push_back(0);
    for (int Q = 0; Q < auxiliary_->nshell(); Q++) {
        int nQ = auxiliary_->shell(Q).nfunction();
        if (counter + nQ > max_rows) {
            counter = 0;
            block_Q_starts.push_back(Q);
        }
        counter += nQ;
    }
    block_Q_starts.push_back(auxiliary_->nshell());

    SharedMatrix Amn2(new Matrix("(A|mn) Block", max_rows, ntri));
    double** Amn2p = Amn2->pointer();
    psio_address next_AIA = PSIO_ZERO;

    const std::vector<std::pair<int,int> >& shell_pairs = sieve_->shell_pairs();
    const size_t npairs = shell_pairs.size();

    // Loop over blocks of Qshell
    for (size_t block = 0; block < block_Q_starts.size() - 1; block++) {

        // Block sizing/offsets
        int Qstart = block_Q_starts[block];
        int Qstop  = block_Q_starts[block+1];
        int qoff   = auxiliary_->shell(Qstart).function_index();
        int nrows  = (Qstop == auxiliary_->nshell() ?
                     auxiliary_->nbf() -
                     auxiliary_->shell(Qstart).function_index() :
                     auxiliary_->shell(Qstop).function_index() -
                     auxiliary_->shell(Qstart).function_index());

        // Compute TEI tensor block (A|mn)

        timer_on("JK: (Q|mn)^R");

        #pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (size_t QMN = 0L; QMN < (Qstop - Qstart) * (ULI) npairs; QMN++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            int Q =  QMN / npairs + Qstart;
            int MN = QMN % npairs;

            std::pair<int,int> pair = shell_pairs[MN];
            int M = pair.first;
            int N = pair.second;

            int nq = auxiliary_->shell(Q).nfunction();
            int nm = primary_->shell(M).nfunction();
            int nn = primary_->shell(N).nfunction();

            int sq =  auxiliary_->shell(Q).function_index();
            int sm =  primary_->shell(M).function_index();
            int sn =  primary_->shell(N).function_index();

            eri2[thread]->compute_shell(Q,0,M,N);

            for (int om = 0; om < nm; om++) {
                for (int on = 0; on < nn; on++) {
                    long int m = sm + om;
                    long int n = sn + on;
                    if (m >= n && schwarz_fun_pairs_r[m*(m+1)/2 + n] >= 0) {
                        long int delta = schwarz_fun_pairs_r[m*(m+1)/2 + n];
                        for (int oq = 0; oq < nq; oq++) {
                            Amn2p[sq + oq - qoff][delta] =
                            buffer2[thread][oq * nm * nn + om * nn + on];
                        }
                    }
                }
            }
        }

        timer_off("JK: (Q|mn)^R");

        // Dump block to disk
        timer_on("JK: (Q|mn)^R Write");

        psio_->write(unit_,"Right (Q|w|mn) Integrals",(char*)Amn2p[0],sizeof(double)*nrows*ntri,next_AIA,&next_AIA);

        timer_off("JK: (Q|mn)^R Write");

    }
    Amn2.reset();
    delete[] eri2;

    psio_->write_entry(unit_, "Omega", (char*) &omega_, sizeof(double));
    psio_->close(unit_,1);
}
void DFJK::rebuild_wK_disk()
{
    // Already open
    outfile->Printf( "    Rebuilding (Q|w|mn) Integrals (new omega)\n\n");

    size_t naux = auxiliary_->nbf();

    // ==> Schwarz Indexing <== //
    const std::vector<std::pair<int,int> >& schwarz_fun_pairs = sieve_->function_pairs();
    int ntri = schwarz_fun_pairs.size();
    const std::vector<long int>&  schwarz_fun_pairs_r = sieve_->function_pairs_reverse();

    // ==> Thread setup <== //
    int nthread = 1;
    #ifdef _OPENMP
        nthread = df_ints_num_threads_;
    #endif

    boost::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    boost::shared_ptr<IntegralFactory> rifactory(new IntegralFactory(auxiliary_, zero, primary_, primary_));
    const double **buffer2 = new const double*[nthread];
    boost::shared_ptr<TwoBodyAOInt> *eri2 = new boost::shared_ptr<TwoBodyAOInt>[nthread];
    for (int Q = 0; Q<nthread; Q++) {
        eri2[Q] = boost::shared_ptr<TwoBodyAOInt>(rifactory->erf_eri(omega_));
        buffer2[Q] = eri2[Q]->buffer();
    }

    ULI maxP = auxiliary_->max_function_per_shell();
    ULI max_rows = memory_ / ntri;
    max_rows = (max_rows > naux ? naux : max_rows);
    max_rows = (max_rows < maxP ? maxP : max_rows);

    // Block extents
    std::vector<int> block_Q_starts;
    int counter = 0;
    block_Q_starts.push_back(0);
    for (int Q = 0; Q < auxiliary_->nshell(); Q++) {
        size_t nQ = auxiliary_->shell(Q).nfunction();
        if (counter + nQ > max_rows) {
            counter = 0;
            block_Q_starts.push_back(Q);
        }
        counter += nQ;
    }
    block_Q_starts.push_back(auxiliary_->nshell());

    SharedMatrix Amn2(new Matrix("(A|mn) Block", max_rows, ntri));
    double** Amn2p = Amn2->pointer();
    psio_address next_AIA = PSIO_ZERO;

    const std::vector<std::pair<int,int> >& shell_pairs = sieve_->shell_pairs();
    const size_t npairs = shell_pairs.size();

    // Loop over blocks of Qshell
    for (size_t block = 0; block < block_Q_starts.size() - 1; block++) {

        // Block sizing/offsets
        int Qstart = block_Q_starts[block];
        int Qstop  = block_Q_starts[block+1];
        int qoff   = auxiliary_->shell(Qstart).function_index();
        int nrows  = (Qstop == auxiliary_->nshell() ?
                     auxiliary_->nbf() -
                     auxiliary_->shell(Qstart).function_index() :
                     auxiliary_->shell(Qstop).function_index() -
                     auxiliary_->shell(Qstart).function_index());

        // Compute TEI tensor block (A|mn)

        timer_on("JK: (Q|mn)^R");

        #pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (size_t QMN = 0L; QMN < (Qstop - Qstart) * (ULI) npairs; QMN++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            int Q =  QMN / npairs + Qstart;
            int MN = QMN % npairs;

            std::pair<int,int> pair = shell_pairs[MN];
            int M = pair.first;
            int N = pair.second;

            int nq = auxiliary_->shell(Q).nfunction();
            int nm = primary_->shell(M).nfunction();
            int nn = primary_->shell(N).nfunction();

            int sq =  auxiliary_->shell(Q).function_index();
            int sm =  primary_->shell(M).function_index();
            int sn =  primary_->shell(N).function_index();

            eri2[thread]->compute_shell(Q,0,M,N);

            for (int om = 0; om < nm; om++) {
                for (int on = 0; on < nn; on++) {
                    long int m = sm + om;
                    long int n = sn + on;
                    if (m >= n && schwarz_fun_pairs_r[m*(m+1)/2 + n] >= 0) {
                        long int delta = schwarz_fun_pairs_r[m*(m+1)/2 + n];
                        for (int oq = 0; oq < nq; oq++) {
                            Amn2p[sq + oq - qoff][delta] =
                            buffer2[thread][oq * nm * nn + om * nn + on];
                        }
                    }
                }
            }
        }

        timer_off("JK: (Q|mn)^R");

        // Dump block to disk
        timer_on("JK: (Q|mn)^R Write");

        psio_->write(unit_,"Right (Q|w|mn) Integrals",(char*)Amn2p[0],sizeof(double)*nrows*ntri,next_AIA,&next_AIA);

        timer_off("JK: (Q|mn)^R Write");
    }
    Amn2.reset();
    delete[] eri2;

    psio_->write_entry(unit_, "Omega", (char*) &omega_, sizeof(double));
    // No need to close
}
void DFJK::manage_JK_core()
{
    Timer manage_jk_core_timing;
    for (int Q = 0 ; Q < auxiliary_->nbf(); Q += max_rows_) {
        int naux = (auxiliary_->nbf() - Q <= max_rows_ ? auxiliary_->nbf() - Q : max_rows_);
        if (do_J_) {
            timer_on("JK: J");
            if(sparse_j_)
            {
                Timer sparse_J_time;
                block_J_sparse(&Qmn_->pointer()[Q],naux);
                if(profile_) outfile->Printf("\n block_J_sparse takes %8.8f s with naux of %d out of %d rows", sparse_J_time.get(), naux, auxiliary_->nbf());
            }
            else
            {
                Timer block_J_time;
                block_J(&Qmn_->pointer()[Q], naux);
                if(profile_) outfile->Printf("\n block_J takes %8.8f s with naux of %d out of %d rows", block_J_time.get(), naux, auxiliary_->nbf());
            }
            timer_off("JK: J");
        }
        if (do_K_) {
            timer_on("JK: K");
            if(sparse_k_)
            {
                Timer sparse_K_time;
                block_K_sparse(&Qmn_->pointer()[Q],naux);
                if(profile_) outfile->Printf("\n block_K_sparse takes %8.8f s with naux of %d out of %d rows", sparse_K_time.get(), naux, auxiliary_->nbf());
            }
            else
            {
                Timer block_K_time;
                block_K(&Qmn_->pointer()[Q],naux);
                if(profile_) outfile->Printf("\n block_K takes %8.8f s with naux of %d out of %d rows", block_K_time.get(), naux, auxiliary_->nbf());
            }
            timer_off("JK: K");
        }
    }
    if(profile_) outfile->Printf("\n manage_JK_core takes %8.8f s.", manage_jk_core_timing.get());
}
void DFJK::manage_JK_disk()
{
    int ntri = sieve_->function_pairs().size();
    Qmn_ = SharedMatrix(new Matrix("(Q|mn) Block", max_rows_, ntri));
    psio_->open(unit_,PSIO_OPEN_OLD);
    for (int Q = 0 ; Q < auxiliary_->nbf(); Q += max_rows_) {
        int naux = (auxiliary_->nbf() - Q <= max_rows_ ? auxiliary_->nbf() - Q : max_rows_);
        psio_address addr = psio_get_address(PSIO_ZERO, (Q*(ULI) ntri) * sizeof(double));

        timer_on("JK: (Q|mn) Read");
        psio_->read(unit_,"(Q|mn) Integrals", (char*)(Qmn_->pointer()[0]),sizeof(double)*naux*ntri,addr,&addr);
        timer_off("JK: (Q|mn) Read");

        if (do_J_) {
            timer_on("JK: J");
            Timer compute_block_j;
            if(sparse_j_) 
                block_J_sparse(&Qmn_->pointer()[0],naux);
            else 
                block_J(&Qmn_->pointer()[0],naux);
            timer_off("JK: J");
            std::string sparse_j = (sparse_j_) ? "sparse" : "dense";

            if(profile_) outfile->Printf("\n Compute J(Disk) %s takes %8.4f s on %d blocks", sparse_j.c_str(),compute_block_j.get(), Q);
        }
        if (do_K_) {
            timer_on("JK: K");
            Timer compute_block_k;
            if(sparse_k_) 
                block_K_sparse(&Qmn_->pointer()[0],naux);
            else
                block_K(&Qmn_->pointer()[0],naux);
            timer_off("JK: K");
            std::string sparse_k = (sparse_k_) ? "sparse" : "dense";
            if(profile_) outfile->Printf("\n Compute K(Disk) %s takes %8.4f s on %d blocks", sparse_k.c_str(), compute_block_k.get(), Q);
        }
    }
    psio_->close(unit_,1);
    Qmn_.reset();
}
void DFJK::manage_wK_core()
{
    int max_rows_w = max_rows_ / 2;
    max_rows_w = (max_rows_w < 1 ? 1 : max_rows_w);
    for (int Q = 0 ; Q < auxiliary_->nbf(); Q += max_rows_w) {
        int naux = (auxiliary_->nbf() - Q <= max_rows_w ? auxiliary_->nbf() - Q : max_rows_w);

        timer_on("JK: wK");
        block_wK(&Qlmn_->pointer()[Q],&Qrmn_->pointer()[Q],naux);
        timer_off("JK: wK");
    }
}
void DFJK::manage_wK_disk()
{
    int max_rows_w = max_rows_ / 2;
    max_rows_w = (max_rows_w < 1 ? 1 : max_rows_w);
    int ntri = sieve_->function_pairs().size();
    Qlmn_ = SharedMatrix(new Matrix("(Q|mn) Block", max_rows_w, ntri));
    Qrmn_ = SharedMatrix(new Matrix("(Q|mn) Block", max_rows_w, ntri));
    psio_->open(unit_,PSIO_OPEN_OLD);
    for (int Q = 0 ; Q < auxiliary_->nbf(); Q += max_rows_w) {
        int naux = (auxiliary_->nbf() - Q <= max_rows_w ? auxiliary_->nbf() - Q : max_rows_w);
        psio_address addr = psio_get_address(PSIO_ZERO, (Q*(ULI) ntri) * sizeof(double));

        timer_on("JK: (Q|mn)^L Read");
        psio_->read(unit_,"Left (Q|w|mn) Integrals", (char*)(Qlmn_->pointer()[0]),sizeof(double)*naux*ntri,addr,&addr);
        timer_off("JK: (Q|mn)^L Read");

        addr = psio_get_address(PSIO_ZERO, (Q*(ULI) ntri) * sizeof(double));

        timer_on("JK: (Q|mn)^R Read");
        psio_->read(unit_,"Right (Q|w|mn) Integrals", (char*)(Qrmn_->pointer()[0]),sizeof(double)*naux*ntri,addr,&addr);
        timer_off("JK: (Q|mn)^R Read");

        timer_on("JK: wK");
        block_wK(&Qlmn_->pointer()[0],&Qrmn_->pointer()[0],naux);
        timer_off("JK: wK");
    }
    psio_->close(unit_,1);
    Qlmn_.reset();
    Qrmn_.reset();
}
void DFJK::block_J(double** Qmnp, int naux)
{
    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    unsigned long int num_nm = function_pairs.size();

    for (size_t N = 0; N < J_ao_.size(); N++) {

        double** Dp   = D_ao_[N]->pointer();
        double** Jp   = J_ao_[N]->pointer();
        double*  J2p  = J_temp_->pointer();
        double*  D2p  = D_temp_->pointer();
        double*  dp   = d_temp_->pointer();
        for (unsigned long int mn = 0; mn < num_nm; ++mn) {
            int m = function_pairs[mn].first;
            int n = function_pairs[mn].second;
            D2p[mn] = (m == n ? Dp[m][n] : Dp[m][n] + Dp[n][m]);
        }
        timer_on("JK: J1");
        C_DGEMV('N',naux,num_nm,1.0,Qmnp[0],num_nm,D2p,1,0.0,dp,1);
        timer_off("JK: J1");

        timer_on("JK: J2");
        C_DGEMV('T',naux,num_nm,1.0,Qmnp[0],num_nm,dp,1,0.0,J2p,1);
        timer_off("JK: J2");
        for (unsigned long int mn = 0; mn < num_nm; ++mn) {
            int m = function_pairs[mn].first;
            int n = function_pairs[mn].second;
            Jp[m][n] += J2p[mn];
            Jp[n][m] += (m == n ? 0.0 : J2p[mn]);
        }
    }
}
void DFJK::block_J_sparse(double ** Qmnp, int naux)
{
    outfile->Printf("\n Computing block_J_sparse using CTF");
    CTF::World comm(MPI_COMM_WORLD);
    int np      = comm.np;
    if(np > 1)
        throw PSIEXCEPTION("Sparse J is not be used in parallel");
    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    unsigned long int num_nm = function_pairs.size();
    int nso = primary_->nbf();

    int three_size[3] = {nso, nso, naux};
    int three_sym[3]  = {NS, NS, NS};
    int two_size[2] = {nso, nso};
    int two_sym[2]  = {NS, NS};
    CTF::Tensor<double> Quv_ctf(3, false, three_size, three_sym);
    CTF::Tensor<double> D_ao(2, false, two_size, two_sym);
    CTF::Tensor<double> J_ao(2, false, two_size, two_sym);
    CTF::Vector<double> J_V(naux);
    int64_t quv_size;
    int64_t* quv_index;
    double* quv_values;
    Quv_ctf.read_local(&quv_size, &quv_index, &quv_values);
    for(int q = 0; q < naux; q++)
    {
        for(int mn = 0; mn < num_nm; mn++)
        {
            int m = function_pairs[mn].first;
            int n = function_pairs[mn].second;
            quv_values[q * nso * nso + m * nso + n ] = Qmnp[q][mn];
            quv_values[q * nso * nso + n * nso + m ] = Qmnp[q][mn];
        }

    }
    //quv_values = Qmnp[0];
    Quv_ctf.write(quv_size, quv_index, quv_values);
    free(quv_index);
    free(quv_values);
    Quv_ctf.sparsify(sparsity_tol_);
    if(profile_) check_sparsity(Quv_ctf, three_size, 3);

    int64_t D_right_size;
    int64_t* D_index;
    double* D_right_values;

    for (size_t N = 0; N < J_ao_.size(); N++)
    {
        double** Dp = D_ao_[N]->pointer();
        D_ao.read_local(&D_right_size, &D_index,&D_right_values);
        C_DCOPY(nso * nso, Dp[0], 1, D_right_values, 1);
        D_ao.write(D_right_size, D_index, D_right_values);
        D_ao.sparsify(sparsity_tol_);
        if(profile_) check_sparsity(D_ao, two_size, 2);
        ///J_Q = B^Q_{pq} D_{pq}
        Timer B_D;
        J_V["Q"] = Quv_ctf["uvQ"] * D_ao["uv"];
        if(profile_) outfile->Printf("\n (B^{Q}_{uv} * D) sparse %8.8f s", B_D.get());
        Timer B_J;
        J_ao["uv"] = Quv_ctf["uvQ"] * J_V["Q"];
        if(profile_) outfile->Printf("\n (B^{Q}_{uv} * J_v(Q)) sparse %8.8f s", B_J.get());
        if(profile_) check_sparsity(J_ao, two_size, 2);
        J_ao.read_local(&D_right_size, &D_index,&D_right_values);

        for(int j = 0; j < D_right_size; j++)
        {
            int m = j / nso;
            int n = j % nso;
            J_ao_[N]->add(m, n, D_right_values[j]);
        }
    }
    free(D_right_values);
    free(D_index);
}
void DFJK::block_K_sparse(double** Qmnp, int naux)
{
    /// K_{uv} = D_{pq} B^{Q}_{up} * B^{Q}_{vq}
    /// KPH wants to use Cyclops to perform a sparse tensor contraction
    /// 
    /// Step 1:  Use def of D = \sum_{i} C_{pi}C_{qi}
    /// Step 2:  Perform a one index transform  (N^4) Use of C_pi should be sparse
    ///          B^{Q}_{ui} = C_{pi} B^{Q}_{up}
    ///          B^{Q}_{vi} = C_{qi} B^{Q}_{vq}
    /// Step 3:  Compute K_{uv} = \sum_{Q} \sum_{i} B^{Q}_{ui} B^{Q}_{vi}

    /// The first iteration of this job will assume that B tensor is distributed 
    /// via the Q index.
    /// This means that all of these steps will be performed locally for every process
    /// Only communciation required will be an Allreduce once the K matrix is formed
    /// GA is used, but we will only perform local MM (so we use data on each processor only)

    /// Can have multiple exchange matrices
    /// GA Specific information
    outfile->Printf("\n Performing a Sparse Exchange build with 1 processors and %d threads", omp_get_max_threads());
    CTF::World comm(MPI_COMM_WORLD);
    int np      = comm.np;
    if(np > 1)
        throw PSIEXCEPTION("Sparse K is not be used in parallel");

    size_t K_size = K_ao_.size();
    Timer Compute_K_all;
    int nbf  = primary_->nbf();
    int three_size[3] = {nbf, nbf, naux};
    int three_sym[3] = {NS, NS, NS};
    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    unsigned long int num_nm = function_pairs.size();

    std::vector<std::string> local_tests;
    if(sparse_type_ == "ALL")
    {
        local_tests.push_back("NORMAL");
        local_tests.push_back("LOCALIZE");
        local_tests.push_back("CHOLESKY");
        local_tests.push_back("DENSITY");
        local_tests.push_back("CHOLESKY_DENSITY");
    }
    else {
        local_tests.push_back(sparse_type_);
    }

    CTF::Tensor<double> Quv_ctf(3, false, three_size, three_sym);
    Quv_ctf.set_name("QUV");
    int64_t quv_size;
    int64_t* quv_index;
    double* quv_values;
    Timer read_local_quv;
    Quv_ctf.read_local(&quv_size, &quv_index, &quv_values);
    if(profile_) outfile->Printf("\n Quv_ctf read local %8.8f s with %d elements", read_local_quv.get(), quv_size);
    #pragma omp parallel for schedule(static)
    for(int q = 0; q < naux; q++)
    {
        for(int mn = 0; mn < num_nm; mn++)
        {
            int m = function_pairs[mn].first;
            int n = function_pairs[mn].second;
            quv_values[q * nbf * nbf + m * nbf + n ] = Qmnp[q][mn];
            quv_values[q * nbf * nbf + n * nbf + m ] = Qmnp[q][mn];
        }

    }
    //quv_values = Qmnp[0];
    Quv_ctf.write(quv_size, quv_index, quv_values);
    Quv_ctf.sparsify(sparsity_tol_);
    check_sparsity(Quv_ctf, three_size, 3);

    for(size_t N = 0; N < K_size; N++)
    {
        int nocc = C_right_ao_[N]->colspi()[0];
        if(not nocc) continue; ///If no occupied orbitals skip exchange
        int q_ui_size[3] = {nbf, nocc, naux};
        int Cui_size[2]  = {nbf, nocc};
        int K_size[2]    = {nbf, nbf};
        int sym_2[2]     = {NS, NS};
        int sym_3[3]     = {NS, NS, NS};

        CTF::Tensor<double> Q_ui(3, false, q_ui_size, sym_3);
        CTF::Tensor<double> Q_uj(3, false, q_ui_size, sym_3);
        CTF::Tensor<double> Q_ur(3, false, three_size, sym_3);
        CTF::Tensor<double> C_right(2, false, Cui_size, sym_2);
        CTF::Tensor<double> C_left(2, false, Cui_size, sym_2);
        CTF::Tensor<double> D_right(2, false, K_size, sym_2);
        CTF::Tensor<double> K(2, false, K_size, sym_2);
        int64_t C_left_size;
        int64_t C_right_size;
        double* C_right_values;
        double* C_left_values;
        int64_t* C_index_left;
        int64_t* C_index_right;

        C_left.read_local(&C_left_size, &C_index_left, &C_left_values);
        C_right.read_local(&C_right_size, &C_index_right, &C_right_values);
        for(int orbital_type = 0; orbital_type < local_tests.size(); orbital_type++)
        {
            Timer sparse_k_type;
            SharedMatrix C_left_matrix(new Matrix("C_left", nbf, nocc));
            SharedMatrix C_right_matrix(new Matrix("C_left", nbf, nocc));
            SharedMatrix local_K(new Matrix("block_k", nbf, nbf));
            C_left.set_zero();
            C_right.set_zero();
            bool c_left_is_c_right = (C_left_ao_[N]->rms() == C_right_ao_[N]->rms());
            if(!c_left_is_c_right)
            {
                outfile->Printf("\n Switching exchange algorithm to NORMAL because C_left and C_right are not symmetric");
            }
            else {
                outfile->Printf("\n Performing Exchange build with %s orbitals", local_tests[orbital_type].c_str());
            }
            Timer get_c_matrix;
            if(local_tests[orbital_type] == "NORMAL")
            {
                C_left_matrix->copy(C_left_ao_[N]);
                C_right_matrix->copy(C_right_ao_[N]);
            }
            else if (local_tests[orbital_type] == "CHOLESKY")
            {
                C_left_matrix->zero();
                C_right_matrix->zero();
                if(c_left_is_c_right)
                {
                    Choleskify(C_left_ao_[N], C_left_matrix, "CHOLESKY_LOCAL");
                    C_right_matrix->copy(C_left_matrix);
                }
                else
                {
                    C_left_matrix->copy(C_left_ao_[N]);
                    C_right_matrix->copy(C_right_ao_[N]);
                }
            }
            else if (local_tests[orbital_type] == "LOCALIZE")
            {
                C_left_matrix->zero();
                C_right_matrix->zero();
                if(c_left_is_c_right)
                {
                    Localize_Occupied(C_left_ao_[N], C_left_matrix);
                    Localize_Occupied(C_right_ao_[N], C_right_matrix);
                }
                else
                {
                    C_left_matrix->copy(C_left_ao_[N]);
                    C_right_matrix->copy(C_right_ao_[N]);
                }
            }
            else if (local_tests[orbital_type] == "DENSITY")
            {
                C_left_matrix->zero();
                C_right_matrix->zero();
                //C_right_matrix->copy(D_ao_[N]);

                int64_t D_right_size;
                int64_t* D_right_index;
                double* D_right_values;

                D_right.read_local(&D_right_size, &D_right_index,&D_right_values);
                int64_t* D_index = new int64_t[D_right_size];
                SharedMatrix D_right_matrix(new Matrix("D_AO", nbf, nbf));
                D_right_matrix->copy(D_ao_[N]);
                Fill_C_Matrices(D_right_size, D_right_values, D_right_matrix);
                D_right.write(D_right_size, D_index, D_right_values);
                free(D_right_values);
                free(D_right_index);
                D_right.sparsify(sparsity_tol_);
            }
            else if (local_tests[orbital_type] == "CHOLESKY_DENSITY")
            {
                C_left_matrix->zero();
                C_right_matrix->zero();
                Choleskify(D_ao_[N], C_left_matrix, "CHOLESKY_DENSITY");
                C_right_matrix->copy(C_left_matrix);
            }
            if(profile_) outfile->Printf("\n Get C Matrix takes %8.5f s.", get_c_matrix.get());

            if(local_tests[orbital_type] != "DENSITY")
            {
                Fill_C_Matrices(C_left_size, C_left_values, C_left_matrix);
                Fill_C_Matrices(C_right_size, C_right_values, C_right_matrix);
                C_left.write(C_left_size  , C_index_left, C_left_values);
                C_right.write(C_right_size, C_index_left, C_right_values);
                C_left.sparsify(sparsity_tol_);
                C_right.sparsify(sparsity_tol_);
            }


            if(local_tests[orbital_type] == "DENSITY")
            {
                //check_sparsity(C_left, Cui_size, 2);
                check_sparsity(D_right, K_size, 2);
            }
            else {
                check_sparsity(C_left, Cui_size, 2);
                check_sparsity(C_right,Cui_size, 2);
            }

            Q_ui.set_zero();
            Q_uj.set_zero();
            Q_ur.set_zero();
            K.set_zero();
            if(local_tests[orbital_type] == "DENSITY")
            {
                Timer Q_ui_sparse;
                Q_ur["uiQ"] = Quv_ctf["urQ"] * D_right["ri"];
                outfile->Printf("\n Quv_ctf * D_right takes %8.4f s.", Q_ui_sparse.get());
                Timer K_sparse;
                K["uv"] = Quv_ctf["urQ"] * Q_ur["vrQ"];
                if(profile_) outfile->Printf("\n K_uv = Q_ui * Q_uj takes %8.4f s.", K_sparse.get());
                check_sparsity(Q_ur, three_size, 3);
            }
             else {
                Timer Q_ui_sparse;
                Q_ui["uiQ"] = Quv_ctf["uvQ"] * C_left["vi"];
                if(profile_) outfile->Printf("\n Quv_ctf * C_right takes %8.4f s.", Q_ui_sparse.get());
                Timer Q_uj_sparse;
                Q_uj["ujQ"] = Quv_ctf["uvQ"] * C_right["vj"];
                outfile->Printf("\n Quv_ctf * C_left takes %8.4f s.", Q_uj_sparse.get());
                Timer K_sparse;
                K["uv"] = Q_ui["uiQ"] * Q_uj["viQ"];
                outfile->Printf("\n K_uv = Q_ui * Q_uj takes %8.4f s.", K_sparse.get());
                check_sparsity(Q_ui, q_ui_size, 3);
                check_sparsity(Q_uj, q_ui_size, 3);
            }
            check_sparsity(K, K_size, 2);
            int64_t k_size;
            int64_t* k_index;
            double* k_values;
            K.read_local(&k_size, &k_index, &k_values);
            C_DCOPY(nbf * nbf, k_values, 1, local_K->pointer()[0], 1);
            if(orbital_type == local_tests.size() - 1)
                C_DAXPY(nbf * nbf, 1.0, local_K->pointer()[0], 1, K_ao_[N]->pointer()[0], 1);
            free(k_values);
            free(k_index);
            if(profile_) outfile->Printf("\n local_K_ao_rms(%s): %8.8f", local_tests[orbital_type].c_str(), local_K->rms());
            if(profile_) outfile->Printf("\n Computing sparse exchange takes %8.5f s with %s.", sparse_k_type.get(), local_tests[orbital_type].c_str());
        }
        free(C_right_values);
        free(C_left_values);
        free(C_index_left);
        free(C_index_right);

        outfile->Printf("\n Overall K_ao_rms = %8.8f", K_ao_[N]->rms());
    }
}
void DFJK::block_K(double** Qmnp, int naux)
{
    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    const std::vector<long int>& function_pairs_reverse = sieve_->function_pairs_reverse();
    unsigned long int num_nm = function_pairs.size();

    for (size_t N = 0; N < K_ao_.size(); N++) {

        Timer single_k;
        int nbf = C_left_ao_[N]->rowspi()[0];
        int nocc = C_left_ao_[N]->colspi()[0];

        if (!nocc) continue;

        double** Clp  = C_left_ao_[N]->pointer();
        double** Crp  = C_right_ao_[N]->pointer();
        double** Elp  = E_left_->pointer();
        double** Erp  = E_right_->pointer();
        double** Kp   = K_ao_[N]->pointer();

        if (N == 0 || C_left_[N].get() != C_left_[N-1].get()) {

            timer_on("JK: K1");

            Timer c_q_e;
            #pragma omp parallel for schedule (dynamic)
            for (int m = 0; m < nbf; m++) {

                int thread = 0;
                #ifdef _OPENMP
                    thread = omp_get_thread_num();
                #endif

                double** Ctp = C_temp_[thread]->pointer();
                double** QSp = Q_temp_[thread]->pointer();

                const std::vector<int>& pairs = sieve_->function_to_function()[m];
                int rows = pairs.size();

                for (int i = 0; i < rows; i++) {
                    int n = pairs[i];
                    long int ij = function_pairs_reverse[(m >= n ? (m * (m + 1L) >> 1) + n : (n * (n + 1L) >> 1) + m)];
                    C_DCOPY(naux,&Qmnp[0][ij],num_nm,&QSp[0][i],nbf);
                    C_DCOPY(nocc,Clp[n],1,&Ctp[0][i],nbf);
                }
                C_DGEMM('N','T',nocc,naux,rows,1.0,Ctp[0],nbf,QSp[0],nbf,0.0,&Elp[0][m*(ULI)nocc*naux],naux);
            }
            if(profile_) outfile->Printf("\n K_dense = C_{\mu i} (Q|mu i) takes %8.8f s.", c_q_e.get());

            timer_off("JK: K1");

        }

        if (!lr_symmetric_ && (N == 0 || C_right_[N].get() != C_right_[N-1].get())) {

            if (C_right_[N].get() == C_left_[N].get()) {
                ::memcpy((void*) Erp[0], (void*) Elp[0], sizeof(double) * naux * nocc * nbf);
            } else {

                timer_on("JK: K1");

                #pragma omp parallel for schedule (dynamic)
                for (int m = 0; m < nbf; m++) {

                    int thread = 0;
                    #ifdef _OPENMP
                        thread = omp_get_thread_num();
                    #endif

                    double** Ctp = C_temp_[thread]->pointer();
                    double** QSp = Q_temp_[thread]->pointer();

                    const std::vector<int>& pairs = sieve_->function_to_function()[m];
                    int rows = pairs.size();

                    for (int i = 0; i < rows; i++) {
                        int n = pairs[i];
                        long int ij = function_pairs_reverse[(m >= n ? (m * (m + 1L) >> 1) + n : (n * (n + 1L) >> 1) + m)];
                        C_DCOPY(naux,&Qmnp[0][ij],num_nm,&QSp[0][i],nbf);
                        C_DCOPY(nocc,Crp[n],1,&Ctp[0][i],nbf);
                    }

                    C_DGEMM('N','T',nocc,naux,rows,1.0,Ctp[0],nbf,QSp[0],nbf,0.0,&Erp[0][m*(ULI)nocc*naux],naux);
                }

                timer_off("JK: K1");

            }

        }

        timer_on("JK: K2");
        Timer q_ui;
        C_DGEMM('N','T',nbf,nbf,naux*nocc,1.0,Elp[0],naux*nocc,Erp[0],naux*nocc,1.0,Kp[0],nbf);
        if(profile_) outfile->Printf("\n K_{mu mu} = (Q|ui) * (Q|vi) takes %8.8f s.", q_ui.get());
        timer_off("JK: K2");
        
        if(profile_) outfile->Printf("\n Compute K densely for %8.8f s with %d density", single_k.get(), N + 1);
    }

}
void DFJK::block_wK(double** Qlmnp, double** Qrmnp, int naux)
{
    const std::vector<std::pair<int, int> >& function_pairs = sieve_->function_pairs();
    const std::vector<long int>& function_pairs_reverse = sieve_->function_pairs_reverse();
    unsigned long int num_nm = function_pairs.size();

    for (size_t N = 0; N < wK_ao_.size(); N++) {

        int nbf = C_left_ao_[N]->rowspi()[0];
        int nocc = C_left_ao_[N]->colspi()[0];

        if (!nocc) continue;

        double** Clp  = C_left_ao_[N]->pointer();
        double** Crp  = C_right_ao_[N]->pointer();
        double** Elp  = E_left_->pointer();
        double** Erp  = E_right_->pointer();
        double** wKp   = wK_ao_[N]->pointer();

        if (N == 0 || C_left_[N].get() != C_left_[N-1].get()) {

            timer_on("JK: wK1");

            #pragma omp parallel for schedule (dynamic)
            for (int m = 0; m < nbf; m++) {

                int thread = 0;
                #ifdef _OPENMP
                    thread = omp_get_thread_num();
                #endif

                double** Ctp = C_temp_[thread]->pointer();
                double** QSp = Q_temp_[thread]->pointer();

                const std::vector<int>& pairs = sieve_->function_to_function()[m];
                int rows = pairs.size();

                for (int i = 0; i < rows; i++) {
                    int n = pairs[i];
                    long int ij = function_pairs_reverse[(m >= n ? (m * (m + 1L) >> 1) + n : (n * (n + 1L) >> 1) + m)];
                    C_DCOPY(naux,&Qlmnp[0][ij],num_nm,&QSp[0][i],nbf);
                    C_DCOPY(nocc,Clp[n],1,&Ctp[0][i],nbf);
                }

                C_DGEMM('N','T',nocc,naux,rows,1.0,Ctp[0],nbf,QSp[0],nbf,0.0,&Elp[0][m*(ULI)nocc*naux],naux);
            }

            timer_off("JK: wK1");

        }

        timer_on("JK: wK1");

        #pragma omp parallel for schedule (dynamic)
        for (int m = 0; m < nbf; m++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            double** Ctp = C_temp_[thread]->pointer();
            double** QSp = Q_temp_[thread]->pointer();

            const std::vector<int>& pairs = sieve_->function_to_function()[m];
            int rows = pairs.size();

            for (int i = 0; i < rows; i++) {
                int n = pairs[i];
                long int ij = function_pairs_reverse[(m >= n ? (m * (m + 1L) >> 1) + n : (n * (n + 1L) >> 1) + m)];
                C_DCOPY(naux,&Qrmnp[0][ij],num_nm,&QSp[0][i],nbf);
                C_DCOPY(nocc,Crp[n],1,&Ctp[0][i],nbf);
            }

            C_DGEMM('N','T',nocc,naux,rows,1.0,Ctp[0],nbf,QSp[0],nbf,0.0,&Erp[0][m*(ULI)nocc*naux],naux);
        }

        timer_off("JK: wK1");

        timer_on("JK: wK2");
        C_DGEMM('N','T',nbf,nbf,naux*nocc,1.0,Elp[0],naux*nocc,Erp[0],naux*nocc,1.0,wKp[0],nbf);
        timer_off("JK: wK2");
    }
}
void DFJK::Fill_C_Matrices(int64_t C_size, double* C_data, SharedMatrix C_matrix)
{
        for(int bf = 0; bf < primary_->nbf(); bf++)
            for(int occ = 0; occ < C_matrix->colspi()[0]; occ++){
                C_data[occ * primary_->nbf() + bf] = 0.0;
                C_data[occ * primary_->nbf() + bf] = C_matrix->get(bf, occ);
            }
}
void DFJK::check_sparsity(CTF::Tensor<double>& tensor_data, int* tensor_dim, int dimension)
{

    int non_zeros = 0;
    int64_t  npair;
    int64_t * global_idx;
    double * non_zero_data;
    tensor_data.read_local_nnz(&npair, &global_idx, &non_zero_data);
    size_t total_elements = 1;
    for(int n = 0; n < dimension; n++)
        total_elements *= tensor_dim[n];

    non_zeros = npair;
    outfile->Printf("\n There are %d non-zeros out of %d which is %8.4f percent sparsity", non_zeros, total_elements, ( 1.0 - (non_zeros * 1.0 / total_elements)) * 100.0);
}
void DFJK::Choleskify(SharedMatrix D_in, SharedMatrix C_out, std::string cholesky_type)
{
    SharedMatrix D_copy(D_in);
    Cholesky* cholesky;
    if(cholesky_type == "CHOLESKY_LOCAL")
    {
        cholesky = new CholeskyLocal(D_copy, 0.000001, 1000000000);
    }
    else {
        cholesky = new CholeskyMatrix(D_copy, 0.000001, 1000000000);
    }
    cholesky->choleskify();
    SharedMatrix C_raw = cholesky->L();
    C_out->zero();
    for(int row = 0; row < C_raw->nrow(); row++)
        for(int col = 0; col < C_raw->ncol(); col++)
            C_out->set(col, row, C_raw->get(row, col));

    delete cholesky;
}
void DFJK::Localize_Occupied(SharedMatrix C_in, SharedMatrix C_out)
{
    SharedMatrix C_copy(C_in);
    boost::shared_ptr<Localizer> localizer = Localizer::build("BOYS", primary_, C_copy);
    localizer->localize();
    C_out->copy(localizer->L());
}

}
