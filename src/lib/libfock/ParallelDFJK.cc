#include "jk.h"
#include "../libmints/mints.h"
#include "../libmints/sieve.h"
#include "psifiles.h"
#include "psi4-dec.h"
#include "../lib3index/3index.h"
#include "../libqt/qt.h"
#include <omp.h>
#include <ga.h>
#include <macdecls.h>
#include <mpi.h>
#include <ctf.hpp>
namespace psi {

ParallelDFJK::ParallelDFJK(boost::shared_ptr<BasisSet> primary, boost::shared_ptr<BasisSet> auxiliary) : JK(primary), auxiliary_(auxiliary)
{
    common_init();
}
ParallelDFJK::~ParallelDFJK()
{
}
void ParallelDFJK::common_init()
{
    outfile->Printf("\n ParallelDFJK");
    memory_ = Process::environment.get_memory();
    //(MPI_COMM_WORLD);
    
}
void ParallelDFJK::preiterations()
{
    /// Compute the sieveing object on all nodes
    if (!sieve_)
    {
        sieve_ = boost::shared_ptr<ERISieve>(new ERISieve(primary_, cutoff_));    
    }
    Timer time_qmn;
    compute_qmn();
    outfile->Printf("\n (Q|MN) takes %8.5f s.", time_qmn.get());
}
void ParallelDFJK::compute_JK()
{
    if(do_J_)
    {
        Timer compute_local_j;
        compute_J();
        outfile->Printf("\n computing J takes %8.5f s.", compute_local_j.get());
    }
    if(do_K_)
    {
        Timer compute_local_K;
        compute_K();
        outfile->Printf("\n computing K Dense takes %8.5f s.", compute_local_K.get());
        Timer compute_K_sparse_time;
        compute_K_sparse();
        outfile->Printf("\n computing K Sparse takes %8.5f s.", compute_K_sparse_time.get());
    }
}
void ParallelDFJK::J_one_half()
{
    // Everybody likes them some inverse square root metric, eh?

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    int naux = auxiliary_->nbf();

    boost::shared_ptr<Matrix> J(new Matrix("J", naux, naux));
    double** Jp = J->pointer();

    int dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = naux;
    chunk[0] = -1;
    chunk[1] = -1;
    J_12_GA_ = NGA_Create(C_DBL, 2, dims, (char *)"J_1/2", chunk);
    if(not J_12_GA_)
        throw PSIEXCEPTION("Failure in creating J_^(-1/2) in GA");

    if(GA_Nodeid() == 0)
    {
        boost::shared_ptr<IntegralFactory> Jfactory(new IntegralFactory(auxiliary_, BasisSet::zero_ao_basis_set(), auxiliary_, BasisSet::zero_ao_basis_set()));
        std::vector<boost::shared_ptr<TwoBodyAOInt> > Jeri;
        for (int thread = 0; thread < nthread; thread++) {
            Jeri.push_back(boost::shared_ptr<TwoBodyAOInt>(Jfactory->eri()));
        }

        std::vector<std::pair<int, int> > Jpairs;
        for (int M = 0; M < auxiliary_->nshell(); M++) {
            for (int N = 0; N <= M; N++) {
                Jpairs.push_back(std::pair<int,int>(M,N));
            }
        }
        long int num_Jpairs = Jpairs.size();

        #pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (long int PQ = 0L; PQ < num_Jpairs; PQ++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            std::pair<int,int> pair = Jpairs[PQ];
            int P = pair.first;
            int Q = pair.second;

            Jeri[thread]->compute_shell(P,0,Q,0);

            int np = auxiliary_->shell(P).nfunction();
            int op = auxiliary_->shell(P).function_index();
            int nq = auxiliary_->shell(Q).nfunction();
            int oq = auxiliary_->shell(Q).function_index();

            const double* buffer = Jeri[thread]->buffer();

            for (int p = 0; p < np; p++) {
            for (int q = 0; q < nq; q++) {
                Jp[p + op][q + oq] =
                Jp[q + oq][p + op] =
                    (*buffer++);
            }}
        }
        Jfactory.reset();
        Jeri.clear();

        // > Invert J < //

        J->power(-1.0/2.0, 1e-10);
        outfile->Printf("\n JRMS: %8.8f", J->rms());
        //for(int me = 0; me < GA_Nnodes(); me++)
        //{
        int begin_offset[2];
        int end_offset[2];
        //NGA_Distribution(J_12_GA_, me, begin_offset, end_offset);
        begin_offset[0] = 0;
        begin_offset[1] = 0;
        end_offset[0]   = naux - 1;
        end_offset[1]   = naux - 1;
        int offset = begin_offset[0];
        NGA_Put(J_12_GA_, begin_offset, end_offset, J->pointer()[0], &naux);
       // }
    }

}
void ParallelDFJK::compute_qmn()
{
    CTF::World dw(MPI_COMM_WORLD);
    // > Sizing < //

    size_t nso = primary_->nbf();
    size_t naux = auxiliary_->nbf();

    if(block_size_ == -1) block_size_ = pow(2, 31)  / (nso * nso * 8) ;

    // > Threading < //

    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    // > Row requirements < //

    // (Q|mn)
    size_t per_row = nso * nso;

    // > Maximum number of rows < //

    int test_memory = (memory_ / per_row);
    //max_rows = 3L * auxiliary_->max_function_per_shell(); // Debug
    test_memory = (test_memory > auxiliary_->nbf() ? auxiliary_->nbf() : test_memory);
    int shell_per_process = 0;
    int shell_start = -1;
    int shell_end = -1;
    /// MPI Environment 
    int my_rank = GA_Nodeid();
    int num_proc = GA_Nnodes();

     shell_per_process = auxiliary_->nshell() / num_proc;
     double memory_requirement= 2.0 * naux * nso * nso * 8.0;
     memory_requirement += (8.0 * naux * naux);
     memory_requirement /= (1024.0 * 1024.0 * 1024.0);
     double memory_in_gb      = memory_ / (1024.0 * 1024.0 * 1024.0);
     outfile->Printf("\n ParallelDF needs %8.5f GB out of %8.5f GB", memory_requirement, memory_in_gb);
     outfile->Printf("\n You need %8.4f nodes to fit (Q|MN) on parallel machine", memory_requirement / memory_in_gb);
     outfile->Printf("\n Memory is %8.4f GB per node", memory_requirement / (memory_in_gb * num_proc));
     ///Fuck it.  Just assume that user provided enough memory (or nodes) for now
     shell_per_process = auxiliary_->nshell() / num_proc;

    
    ///Have first proc be from 0 to shell_per_process
    ///Last proc is shell_per_process * my_rank to naux

    int  dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = nso * nso;
    chunk[0] = GA_Nnodes();
    chunk[1] = 1;
    int map[GA_Nnodes() + 1];
    for(int iproc = 0; iproc < GA_Nnodes(); iproc++)
    {
        int p_shell_start = 0;
        int p_shell_end = 0;
        if(iproc != (num_proc - 1))
        {
            p_shell_start = shell_per_process * iproc;
            p_shell_end   = shell_per_process * (iproc + 1);
        }
        else
        {
            p_shell_start = shell_per_process * iproc;
            p_shell_end = (auxiliary_->nshell() % num_proc == 0 ? shell_per_process * (iproc + 1) : auxiliary_->nshell());
        }
        int p_function_start = auxiliary_->shell(p_shell_start).function_index();
        int p_function_end = (p_shell_end == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(p_shell_end).function_index());
        map[iproc] = p_function_start;
        outfile->Printf("\n  P%d shell_start: %d shell_end: %d function_start: %d function_end: %d", iproc, p_shell_start, p_shell_end, p_function_start, p_function_end);
    }
    ///set the shell index to be processor specific
    if(my_rank != (num_proc - 1))
    {
        shell_start = shell_per_process * my_rank;
        shell_end   = shell_per_process * (my_rank + 1);
    }
    else
    {
        shell_start = shell_per_process * my_rank;
        shell_end = (auxiliary_->nshell() % num_proc == 0 ? shell_per_process * (my_rank + 1) : auxiliary_->nshell());
    }

    int function_start = auxiliary_->shell(shell_start).function_index();
    int function_end = (shell_end == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(shell_end).function_index());
    int max_rows = (function_end - function_start);

    printf("\n  P%d shell_start: %d shell_end: %d function_start: %d function_end: %d", GA_Nodeid(), shell_start, shell_end, function_start, function_end);
    printf("\n P%d max_rows: %d", GA_Nodeid(), max_rows);
    map[GA_Nnodes()] = 0;
    outfile->Printf("\n About to create A_UV");
    int A_UV_GA = NGA_Create_irreg(C_DBL, 2, dims, (char *)"Auv_temp", chunk, map);
    if(not A_UV_GA)
    {
        outfile->Printf("\n GA failed to create (A|UV)");
        throw PSIEXCEPTION("GA failed on creating Aia_ga");
    }
    GA_Print_distribution(A_UV_GA);
    Q_UV_GA_ = GA_Duplicate(A_UV_GA, (char *)"Q|PQ");
    if(not Q_UV_GA_)
    {
        outfile->Printf("\n GA failed to create (Q|UV) via duplicate");
        throw PSIEXCEPTION("GA failed on creating GA_Q_PQ");
    }
    outfile->Printf("\n Created (A|UV) and (Q|UV) on P0");
    if(GA_Nodeid() > 1)
        printf("\n Created (A|UV) and (Q|UV) on P%d", GA_Nodeid());

    // => ERI Objects <= //

    boost::shared_ptr<IntegralFactory> factory(new IntegralFactory(auxiliary_, BasisSet::zero_ao_basis_set(), primary_, primary_));
    std::vector<boost::shared_ptr<TwoBodyAOInt> > eri;
    for (int thread = 0; thread < nthread; thread++) {
            eri.push_back(boost::shared_ptr<TwoBodyAOInt>(factory->eri()));
    }

    // => ERI Sieve <= //

    //boost::shared_ptr<ERISieve> sieve(new ERISieve(primary_, 1e-10));
    const std::vector<std::pair<int,int> >& shell_pairs = sieve_->shell_pairs();
    long int nshell_pairs = (long int) shell_pairs.size();

    // => Temporary Tensors <= //

    // > Three-index buffers < //
    //// ==> Master Loop <== //

    int Auv_begin[2];
    int Auv_end[2];
    /// SIMD 
    ///shell_start represents the start of shells for this processor
    ///shell_end represents the end of shells for this processor
    ///NOTE:  This code will have terrible load balance (shells do not correspond to equal number of functions
    //CTF::Matrix<double> Auv_ctf(naux, nso * nso,dw);
    //CTF::Matrix<double> Quv_ctf(naux, nso * nso,dw);
    //CTF::Matrix<double> J_12_ctf(naux, naux,  dw);
    int auv_edge[3] = {naux, nso, nso};
    int j12_edge[2] = {naux, naux};
    CTF::Tensor<double> Auv_ctf(3, false, auv_edge, dw);
    CTF::Tensor<double> Quv_ctf_(3, false, auv_edge, dw);
    CTF::Tensor<double> J_12_ctf(2, false,j12_edge, dw);
    int64_t local_size = max_rows * nso * nso;
    int64_t * local_index = new int64_t[local_size];
    double * local_values = new double[local_size];
    int64_t size;
    int64_t * indices;
    double  * values;

    Timer compute_Auv;
    {
        std::vector<double> Auv(max_rows * nso * nso, 0.0);
        //boost::shared_ptr<Matrix> Auv(new Matrix("(Q|mn)", 8 * max_rows, nso * (unsigned long int) nso));
        //double** Auvp = Auv->pointer();

        int Pstart = shell_start;
        int Pstop  = shell_end;
        int nPshell = Pstop - Pstart;
        int pstart = auxiliary_->shell(Pstart).function_index();
        int pstop = (Pstop == auxiliary_->nshell() ? auxiliary_->nbf() : auxiliary_->shell(Pstop).function_index());
        int rows = pstop - pstart;

        // > (Q|mn) ERIs < //

        //::memset((void*) Auvp[0], '\0', sizeof(double) * rows * nso * nso);
        if(debug_) printf("\n P%d rows: %d (%d - %d)", GA_Nodeid(), rows, pstop, pstart);
        if(debug_) printf("\n P%d maxrows: %d", GA_Nodeid(), max_rows);

        Timer compute_integrals_raw;
        if(profile_) printf("\n P%d about to compute integrals", GA_Nodeid());
        #pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (long int PMN = 0L; PMN < nPshell * nshell_pairs; PMN++) {

            int thread = 0;
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif

            int P  = PMN / nshell_pairs + Pstart;
            int MN = PMN % nshell_pairs;
            std::pair<int,int> pair = shell_pairs[MN];
            int M = pair.first;
            int N = pair.second;

            eri[thread]->compute_shell(P,0,M,N);

            int nm = primary_->shell(M).nfunction();
            int nn = primary_->shell(N).nfunction();
            int np = auxiliary_->shell(P).nfunction();
            int om = primary_->shell(M).function_index();
            int on = primary_->shell(N).function_index();
            int op = auxiliary_->shell(P).function_index();

            const double* buffer = eri[thread]->buffer();

            for (int p = 0; p < np; p++) {
            for (int m = 0; m < nm; m++) {
            for (int n = 0; n < nn; n++) {
                //Auvp[p + op - pstart][(m + om) * nso + (n + on)] =
                //Auvp[p + op - pstart][(n + on) * nso + (m + om)] =
                Auv[(p + op - pstart) * nso * nso  + (m + om) * nso + (n + on)] = 
                Auv[(p + op - pstart) * nso * nso  + (n + on) * nso + (m + om)] = 
                (*buffer++);
            }}}

        }

        //printf("\n P%d size: %d ", GA_Nodeid(), size);
        //for(int64_t index = 0; index < size; index++){
        ////    printf("\nP%d indices[%d]: %d =  %8.8f", GA_Nodeid(),index, indices[index], values);
        //    printf("\n P%d size: %d", GA_Nodeid(), size);
        //}
            
        printf("\n Auv local: local_size: %d", local_size);
        for(int q = function_start; q < function_end; q++)
            for(int u = 0; u < nso; u++)
                for(int v = 0; v < nso; v++)
                {
                    size_t local_offset = u * nso * (max_rows) + v * (max_rows) + (q - function_start); 
                    local_index[local_offset] = u * nso * naux + v * naux + q + function_start;
                    local_values[local_offset] = Auv[(q - function_start) * nso * nso + u * nso + v];
                    //local_values[local_offset] = 0.0;
        //            printf("\n P%d local_index[%d] = %d local_values[%d] = %8.10f", GA_Nodeid(),local_offset, local_index[local_offset], local_offset, local_values[local_offset]);
         //           printf("\n P%d u: %d v: %d naux: %d", GA_Nodeid(), u, v, q + function_start);
                }
        //int64_t   ctf_local_size;
        //int64_t*  ctf_local_index;
        //double*   ctf_local_values;

        //Auv_ctf.read_local(&ctf_local_size, &ctf_local_index, &ctf_local_values);
        //for(size_t me = 0; me < ctf_local_size; me++){
        //    size_t ctf_local = ctf_local_index[me];
        //    size_t d1 = ctf_local % naux;
        //    size_t d2 = (ctf_local / naux) % nso;
        //    size_t d3 = (ctf_local / (nso * naux)) % (nso * naux);
        //    size_t my_offset = d1 + d2 * naux + d3 * naux * nso;
        //    printf("\n P%d ctf_local_index[%d] = %d ctf_my_offset: %d", GA_Nodeid(),me, ctf_local_index[me], my_offset);
        //    printf("\n P%d d1(Q): %d d2(v): %d d3(u): %d", GA_Nodeid(), d1, d2, d3);

        //}

        Auv_ctf.write(local_size, local_index, local_values);
        if(profile_) printf("\n P%d Computing integrals takes %8.4f s.", GA_Nodeid(), compute_integrals_raw.get());
        //NGA_Distribution(A_UV_GA, GA_Nodeid(), Auv_begin, Auv_end);
        //int ld = nso * nso;
        Timer put_ga_auv;
        get_or_put_ga_batches(A_UV_GA, Auv, false);
        if(profile_) printf("\n P%d Placing Auv into Distributed memory takes %8.4f s.", GA_Nodeid(), put_ga_auv.get());
    }
    if(profile_) printf("\n  P%d Auv took %8.6f s.", GA_Nodeid(), compute_Auv.get());

    Timer J_one_half_time;
    J_one_half();
    int begin_offset[2];
    int end_offset[2];
    begin_offset[0] = 0;
    begin_offset[1] = 0;
    end_offset[0] = auxiliary_->nbf() - 1;
    end_offset[1] = auxiliary_->nbf() - 1;
    std::vector<double> j_12_buf(naux * naux);
    int stride = naux;
    NGA_Get(J_12_GA_, begin_offset, end_offset, &j_12_buf[0], &stride);
    if(profile_) printf("\n  P%d J^({-1/2}} took %8.6f s.", GA_Nodeid(), J_one_half_time.get());
    J_12_ctf.read_local(&size,&indices, &values);
    for(int q = 0; q < naux * naux; q++)
        values[q] = j_12_buf[q];
    J_12_ctf.write(size, indices, values);
    free(indices);
    free(values);

    Timer GA_DGEMM;
    GA_Dgemm('N', 'N', naux, nso * nso, naux, 1.0, J_12_GA_, A_UV_GA, 0.0, Q_UV_GA_);
    //GA_Print(J_12_GA_);
    //J_12_ctf.print();

    Quv_ctf_["Quv"] = J_12_ctf["QA"] * Auv_ctf["Auv"];

    double * a_values = new double[local_size];
    Quv_ctf_.read(local_size, local_index, a_values);
    Quv_ctf_.read_local(&size, &indices, &values);
    if(profile_) printf("\n  P%d DGEMM took %8.6f s.", GA_Nodeid(), GA_DGEMM.get());
    if(profile_) outfile->Printf("\n GA_Dgemm takes %8.4f s.", GA_DGEMM.get());
    local_quv_.resize(max_rows * nso * nso);
    Timer get_local_quv;
    //get_or_put_ga_batches(Q_UV_GA_, local_quv_, true);
    for(int q = 0; q < max_rows; q++)
        for(int u = 0; u < nso; u++)
            for(int v = 0; v < nso; v++)
    {
        //outfile->Printf("\n local_quv_: %8.8f values: %8.8f", local_quv_[i], values[i]);
        //outfile->Printf("\n local_values: %8.8f values: %8.8f indices: %d global_index: %d", local_values[u * nso * max_rows + v * max_rows + q], values[u * nso * max_rows + v * max_rows + q], indices[u * nso * max_rows + v * max_rows + q], local_index[u * nso * max_rows + v * max_rows + q]);
        local_quv_[q * nso * nso + u * nso + v] = a_values[u * nso * max_rows + v * max_rows + q];
    }
    free(values);
    free(indices);
    delete[] local_index;
    delete[] local_values;
    delete[] a_values;

    if(profile_) printf("\n P%d Get local_quv_ take %8.4f s.", GA_Nodeid(), get_local_quv.get());
    outfile->Printf("\n Get local_quv takes %8.4f s.", get_local_quv.get());

    Timer destroy_ga_arrays;
    GA_Destroy(A_UV_GA);
    if(profile_) printf("\n P%d Destroyed A_UV takes %8.4f s.", destroy_ga_arrays.get());
    GA_Destroy(J_12_GA_);
    if(profile_) printf("\n P%d Destroyed J_{12} takes %8.4f s.", destroy_ga_arrays.get());
    GA_Destroy(Q_UV_GA_);
    if(profile_) printf("\n P%d Destroyed Q_UV takes %8.4f s.", destroy_ga_arrays.get());
}
void ParallelDFJK::compute_J()
{
    ///Some basic information (naux -> auxiliary basis set size
    int naux = auxiliary_->nbf();
    ///(nso -> number of basis functions
    int nso = D_ao_[0]->rowspi()[0];
    unsigned long int num_nm = nso * nso;

    SharedVector J_temp(new Vector("Jtemp", num_nm));
    SharedVector D_temp(new Vector("Dtemp", num_nm));
    double* D_tempp = D_temp->pointer();

    ///Local q_uv for get and J_V
    std::vector<double> J_V;
    Timer Compute_J_all;

    size_t local_naux = local_quv_.size() / num_nm;
    J_V.resize(local_naux);

    for(size_t N = 0; N < J_ao_.size(); N++)
    {
        Timer Compute_J_one;
        ///This loop is parallelized over MPI
        ///Q_UV_GA is distributed with auxiliary_index
        double** Dp = D_ao_[N]->pointer();
        double** Jp = J_ao_[N]->pointer();
        ///J_Q = B^Q_{pq} D_{pq}
        Timer v_BD;
        //size_t local_naux = end_offset[0] - begin_offset[0] + 1;
        //C_DGEMV('N', local_naux, num_nm, 1.0, &q_uv_temp[0], num_nm, Dp[0], 1, 0.0, &J_V[0], 1);
        C_DGEMV('N', local_naux, num_nm, 1.0, &local_quv_[0], num_nm, Dp[0], 1, 0.0, &J_V[0], 1);
        if(profile_) printf("\n P%d (Q|MN) * D_{MN} takes %8.4f s. ", GA_Nodeid(), v_BD.get());
        double J_V_norm = 0.0;
        Timer J_BJ;
        ///J_{uv} = B^{Q}_{uv} J_V^{Q}
        //C_DGEMV('T', local_naux, num_nm, 1.0, &q_uv_temp[0], num_nm, &J_V[0], 1, 0.0, J_temp->pointer(), 1);
        C_DGEMV('T', local_naux, num_nm, 1.0, &local_quv_[0], num_nm, &J_V[0], 1, 0.0, J_temp->pointer(), 1);

        if(profile_) printf("\n P%d (B^{Q}_{uv} * J_V^{Q} takes %8.4f s", GA_Nodeid(), J_BJ.get());
        ///Since every processor has a copy of J_temp, sum all the parts and send to every process
        Timer all_reduce;
        MPI_Allreduce(J_temp->pointer(), Jp[0], num_nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(profile_) printf("\n P%d Allreduce for J takes %8.4f s.", GA_Nodeid(), all_reduce.get());

        if(profile_) printf("\n P%d Compute_J for %d density takes %8.6f s with ||J||: %8.8f", GA_Nodeid(), Compute_J_one.get(), N, J_temp->norm());
    }
    if(profile_) printf("\nP%d Compute_J takes %8.6f s for %d densities", GA_Nodeid(), Compute_J_all.get(), J_ao_.size());
}
void ParallelDFJK::compute_K()
{
    /// K_{uv} = D_{pq} B^{Q}_{up} * B^{Q}_{vq}
    /// Step 1:  Use def of D = \sum_{i} C_{pi}C_{qi}
    /// Step 2:  Perform a one index transform  (N^4)
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
    int begin_offset[2];
    int end_offset[2];
    int index = 0;
    size_t nso = primary_->nbf();
    size_t stride     = nso * nso;
    size_t local_naux = local_quv_.size() / stride;

    size_t K_size = K_ao_.size();
    Timer Compute_K_all;
    for(size_t N = 0; N < K_size; N++)
    {
        int nbf = C_left_ao_[N]->rowspi()[0];
        int nocc = C_left_ao_[N]->colspi()[0];
        double** Clp = C_left_ao_[N]->pointer();
        double** Crp = C_right_ao_[N]->pointer();
        double** Kp  = K_ao_[N]->pointer();
        SharedMatrix BQ_ui(new Matrix("B^Q_{ui}", local_naux * nbf, nocc));
        SharedMatrix BQ_vi(new Matrix("B^Q_{vi}", local_naux * nbf, nocc));
        SharedMatrix Bm_Qi(new Matrix("B^m_{Qi}", nbf, local_naux * nocc));
        SharedMatrix Bn_Qi(new Matrix("B^n_{Qi}", nbf, local_naux * nocc));
        int count = 0;
        for(int mu = 0; mu < nbf; mu++)
            for(int occ = 0; occ < nocc; occ++)
                if(C_left_ao_[N]->get(mu, occ) > 1e-6) count++;
        //outfile->Printf("\n Absolute sparsity: %d / %d Percentage sparsity (%8.4f )", count, nbf * nocc, count / (nbf * nocc));

        if(not nocc) continue; ///If no occupied orbitals skip exchange

        if(N == 0 or C_left_[N].get() != C_left_[N-1].get())
        {
            Timer B_C_halftrans;

            C_DGEMM('N', 'N', local_naux * nbf, nocc, nbf, 1.0, &local_quv_[0], nbf, Clp[0], nocc, 0.0, BQ_ui->pointer()[0], nocc);
            if(profile_) printf("\n P%d B * C takes %8.4f", GA_Nodeid(), B_C_halftrans.get());
	    if(profile_) outfile->Printf("\n B * C takes %8.4f", B_C_halftrans.get());

            Timer swap_index;
            #pragma omp parallel for
            for(int n = 0; n < local_naux; n++)
                for(int m = 0; m < nbf; m++)
                    C_DCOPY(nocc, &BQ_ui->pointer()[0][n * nbf * nocc + m * nocc], 1, &Bm_Qi->pointer()[0][m * local_naux * nocc + n * nocc], 1);
            if(profile_) printf("\n P%d Bm_Qi to BQ_ui takes %8.4f s.", GA_Nodeid(), swap_index.get());
	    if(profile_) outfile->Printf("\n Bm_Qi to BQ_ui takes %8.4f s.", swap_index.get());

        if(lr_symmetric_)
            Bn_Qi = Bm_Qi;
             
        }
        if(!lr_symmetric_ && (N == 0 || C_right_[N].get() != C_right_[N-1].get())) {

            Timer B_C_righttrans;
            if(C_right_[N].get() == C_left_[N].get()) 
            {
                //::memcpy((void*) Bm_Qi->pointer()[0], (void*) Bn_Qi->pointer()[0], sizeof(double) * local_naux * nocc * nbf);
                Bn_Qi = Bm_Qi;
            }
            else {
            C_DGEMM('N', 'N',local_naux * nbf, nocc, nbf, 1.0, &local_quv_[0], nbf, Crp[0], nocc, 0.0, BQ_ui->pointer()[0], nocc);

            #pragma omp parallel for
            for(int n = 0; n < local_naux; n++)
                for(int m = 0; m < nbf; m++)
                    C_DCOPY(nocc, &BQ_ui->pointer()[0][n * nbf * nocc + m * nocc], 1, &Bn_Qi->pointer()[0][m * local_naux * nocc + n * nocc], 1);

            }
            if(profile_) printf("\n P%d Half trans right side takes %8.4f s", GA_Nodeid(), B_C_righttrans.get()); 
	    if(profile_) outfile->Printf("\n Half trans right size takes %8.4f s.", B_C_righttrans.get());
        }
        SharedMatrix local_K(new Matrix("K", nbf, nbf));
        Timer Final_K;
        C_DGEMM('N','T', nbf, nbf, local_naux * nocc, 1.0, Bm_Qi->pointer()[0], local_naux * nocc, Bn_Qi->pointer()[0], local_naux * nocc, 0.0, local_K->pointer()[0], nbf);
        if(profile_) printf("\n P%d Final_K takes %8.4f s and norm is %8.4f", GA_Nodeid(), Final_K.get(), local_K->rms());
        if(profile_) outfile->Printf("\nFinal_K takes %8.4f s and norm is %8.4f", Final_K.get(), local_K->rms());
        
        Timer ALLREDUCE;
        MPI_Allreduce(local_K->pointer()[0],Kp[0], nbf * nbf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(profile_) printf("\n P%d ALLREDUCE takes %8.4f s", GA_Nodeid(), ALLREDUCE.get());
	if(profile_) outfile->Printf("\n ALLREDUCE takes %8.4f s.", ALLREDUCE.get());
        
    }
    printf("\n P%d for Compute K with %d densities %8.4f s", GA_Nodeid(), D_ao_.size(), Compute_K_all.get());
}
void ParallelDFJK::compute_K_sparse()
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
    CTF::World dw(MPI_COMM_WORLD);
    size_t K_size = K_ao_.size();
    Timer Compute_K_all;
    int naux = auxiliary_->nbf();
    int nbf  = primary_->nbf();
    int64_t quv_ctf_size;
    int64_t* quv_index;
    double* quv_ctf_values;
    int aux_edge[3] = {naux, nbf, nbf};
    CTF::Tensor<double> Quv_ctf(3, true, aux_edge, dw);
    Quv_ctf.read_local(&quv_ctf_size, &quv_index,&quv_ctf_values);
    int nso = nbf;
    for(int q = 0; q < naux; q++)
        for(int u = 0; u < nbf; u++)
            for(int v = 0; v < nbf; v++)
    {
        //outfile->Printf("\n local_quv_: %8.8f values: %8.8f", local_quv_[i], values[i]);
        quv_ctf_values[u * nso * naux + v * naux + q] = local_quv_[q * nso * nso + u * nso + v];
    }
    Quv_ctf.write(quv_ctf_size, quv_index, quv_ctf_values);

    for(size_t N = 0; N < K_size; N++)
    {   
        int nocc = C_right_ao_[N]->colspi()[0];
        if(not nocc) continue; ///If no occupied orbitals skip exchange
        int q_ui_size[3] = {naux, nbf, nocc};
        int Cui_size[2]  = {nbf, nocc};
        int K_size[2]    = {nbf, nbf};
        int sym_2[2]     = {NS, NS};
        int sym_3[3]     = {NS, NS, NS};
        
        CTF::Tensor<double> Q_ui(3, false, q_ui_size, sym_3);
        CTF::Tensor<double> Q_uj(3, false, q_ui_size, sym_3);
        CTF::Tensor<double> C_right(2, false, Cui_size, sym_2);
        CTF::Tensor<double> C_left(2, false, Cui_size, sym_2);
        CTF::Tensor<double> K(2, false, K_size, sym_2);
        int64_t C_left_size;
        double* C_left_values;
        int64_t C_right_size;
        double* C_right_values;
        int64_t* C_index = new int64_t[nbf * nocc];

        for(int mu = 0; mu < nbf; mu++)
            for(int occ = 0; occ < nocc; occ++)
                C_index[occ * nbf + mu] = occ * nbf + mu;
        C_left.read_all(&C_left_size, &C_left_values);
        C_right.read_all(&C_right_size, &C_right_values);
        SharedMatrix C_left_matrix(new Matrix("C_left", nbf, nocc));
        SharedMatrix C_right_matrix(new Matrix("C_left", nbf, nocc));
        C_left_matrix = C_left_ao_[N];
        C_right_matrix = C_right_ao_[N];
        Fill_C_Matrices(C_left_size, C_left_values, C_left_matrix);
        Fill_C_Matrices(C_right_size, C_right_values, C_right_matrix);
        C_left.write(C_left_size, C_index,C_left_values);
        C_right.write(C_right_size, C_index,C_right_values);
        C_left.sparsify(1e-10);
        C_right.sparsify(1e-10);
        Quv_ctf.sparsify(1e-10);

        //fill_c_matrices_ctf(C_right)
        //Quv_ctf.print();
        //C_right.print();
        C_right.print(stdout, 1e-64);
        C_right_matrix->print();
        Q_ui["Qui"] += Quv_ctf["Quv"] * C_right["vi"];
        Q_uj["Quj"] += Quv_ctf["Quv"] * C_left["vj"];
        K["uv"] = Q_ui["Qui"] * Q_uj["Qvi"];
        int64_t k_size;
        double* k_values;
        
        K.read_all(&k_size, &k_values);
        for(int mu = 0; mu < nbf; mu++)
            for(int nu = 0; nu < nbf; nu++)
                K_ao_[N]->set(mu, nu, k_values[mu * nbf + nu]);
        free(k_values);
        delete[] C_index;
    }
    printf("\n P%d for Compute K with %d densities %8.4f s", GA_Nodeid(), D_ao_.size(), Compute_K_all.get());
}

void ParallelDFJK::postiterations()
{
}
void ParallelDFJK::print_header() const
{
    outfile->Printf("\n Computing DFJK using %d Processes and %d threads", GA_Nnodes(), omp_get_max_threads());
}
void ParallelDFJK::get_or_put_ga_batches(int MY_GA, std::vector<double>& ga_buf, bool ga_get)
{
    int begin_offset[2];
    int end_offset[2];
    NGA_Distribution(MY_GA, GA_Nodeid(), begin_offset, end_offset);
    int stride = end_offset[1] - begin_offset[1] + 1;
    if(end_offset[0] - begin_offset[0] < block_size_)
    {   
        //printf("\n GA_GET: %d offset[0] = (%d, %d)", ga_get, begin_offset[0], end_offset[0]);
        if(ga_get)
            NGA_Get(MY_GA, begin_offset, end_offset, &ga_buf[0], &stride);
        else
            NGA_Put(MY_GA, begin_offset, end_offset, &ga_buf[0], &stride);

    }
    else {
        ///Since I am here, I need to read in batches of naux. 
        int naux_batch = (end_offset[0] - begin_offset[0] + 1) / block_size_;
        int naux_begin =  begin_offset[0];
        int naux_end   =  end_offset[0];
        for(int batch = 0; batch < naux_batch; batch++)
        {   
            int begin = naux_begin + (batch * block_size_);
            int end   = (batch == naux_batch - 1) ? naux_end : naux_begin + (batch + 1) * block_size_ - 1;
            begin_offset[0] = begin;
            end_offset[0] = end;
            //printf("\n P%d offset[%d][0] = (%d, %d) offset[%d][1] = (%d, %d) and ld: %d", GA_Nodeid(), batch, begin_offset[0], end_offset[0], batch, begin_offset[1], end_offset[1], stride);
            if(ga_get)
                NGA_Get(MY_GA, begin_offset, end_offset, &ga_buf[(batch * block_size_) * stride], &stride);
            else
                NGA_Put(MY_GA, begin_offset, end_offset, &ga_buf[(batch * block_size_) * stride], &stride);
        }
    }
    double ga_buf_norm = 0.0;
    //for(auto value : ga_buf)
    //    ga_buf_norm += value * value;
    //printf("\n Norm in get_ga: %8.8f", sqrt(ga_buf_norm));
}
//void ParallelDFJK::DF_Dgemm(int GA_left, int GA_right, int GA_final)
//{
//    if(GA_Nnodes() == 0)
//    {
//        size_t naux = auxiliary_->nbf();
//        size_t nso2 = primary_->nbf() * primary_->nbf();
//
//        int begin_j[2];
//        int end_j[2];
//        int begin_qmn[2];
//        int end_qmn[2];
//
//        NGA_Distribution(GA_left, 0, begin_j, end_j);
//        std::vector<double> j_buf(end_j[0] - begin_j[0] + 1, 0.0);
//        get_or_put_ga_batches(GA_left, j_buf, true);
//
//        NGA_Distribution(GA_right, 0, begin_qmn, end_qmn);
//        std::vector<double> qmn_buf(nso2 * naux, 0.0);
//        std::vector<double> final_buf(qmn_buf);
//        get_or_put_ga_batches(GA_right, qmn_buf, true);
//
//        C_DGEMM('N', 'N', naux, nso2, naux, 1.0, &j_buf[0], naux, &qmn_buf[0], nso2, 0.0, &final_buf[0], nso2);
//
//        get_or_put_ga_batches(GA_final, final_buf, false);
//
//    }
//    else {
//        int naux = auxiliary_->nbf();
//        int nso = primary_->nbf();
//        GA_Dgemm('N', 'N', naux, nso * nso, naux, 1.0, GA_left, GA_right, 0.0, GA_final);
//    }
//}
void ParallelDFJK::Fill_C_Matrices(int64_t C_size, double* C_data, SharedMatrix C_matrix)
{
    for(int bf = 0; bf < primary_->nbf(); bf++)
        for(int occ = 0; occ < C_matrix->colspi()[0]; occ++){
            C_data[occ * primary_->nbf() + bf] = C_matrix->get(bf, occ);
        }
}
}
