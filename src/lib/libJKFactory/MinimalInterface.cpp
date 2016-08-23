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
#include <mpi.h>
#include <ga.h>
#include <macdecls.h>
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
   Comm_->FreeComm();
   PFock_destroy(PFock_);
   CInt_destroyBasisSet(GTBasis_);
}

psi::MinimalInterface::MinimalInterface(const int NMats,
      const bool AreSymm):NPRow_(1),NPCol_(1),
                          StartRow_(0),StartCol_(0),
                          EndRow_(0),EndCol_(0),Stride_(0),NBasis_(0){
    //Trying to parallelize over the density matrices
    //Try to split up communicators
    ///How many density matrices to have GTFock be responsible for
    ///If user does not specify this, assume GTFock handles all density matrices
    psi::Options& options = psi::Process::environment.options;
    GlobalComm_ = psi::WorldComm->GetComm();
    int total_number_processors = GlobalComm_->NProc();
    int global_me = GlobalComm_->Me();
    outfile->Printf("\n Using GTFock for ParallelJK build with %d MPI processes and %d omp threads", total_number_processors, omp_get_max_threads());
    int density_matrices_per_process = options.get_int("DENSITY_MATRICES_PER_PROCESS");
    int subgroup_number              = options.get_int("NUMBER_OF_SUBGROUPS");
    ///Create a subgroup that will process density_matrices_per_process
    create_communicators(NMats, density_matrices_per_process, subgroup_number);
    SetUp();
    SplitProcs(NPRow_,NPCol_);
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
   /// This will tell which subgroup I am on
   int which_processor_group = (global_me % subgroup_);
   ///Since subgroup_ starts from 1, need to increment by 1
   size_t my_group_size = subgroup_to_density_[which_processor_group + 1].size();
   std::vector<int> processor_list;
   int processor_size = 0;
   create_processor_list(processor_list, processor_size, NMats);
   int return_flag = (int) PFock_create(GTBasis_,NPRow_,NPCol_,NBlkFock,IntThresh,
         static_cast<int>(my_group_size) ,AreSymm,&PFock_, &processor_list[0], processor_size);

   if(return_flag != 0)
   {
       for(int i = 0; i < processor_size; i++)
           printf("\n P%d processor_list[%d] = %d", global_me, i, processor_list[i]);
       outfile->Printf("\n Something happened during PFock_create"); 
       outfile->Printf("\n GTFock error is %d", return_flag);
       outfile->Printf("\n NMats: %d AreSymm: %d", NMats, AreSymm);
       throw PSIEXCEPTION("GTFock threw a failure in PFock_create. Check error and log.");
   }
   printf("\n PFock_create is complete.  Took %8.6f s. ", pfock_create.get());
}

void psi::MinimalInterface::SetP(std::vector<SharedMatrix>& Ps){
   int return_flag = 0;
   int which_processor_group = (GlobalComm_->Me() % subgroup_ ) + 1;
   std::vector<int> my_density_chunk = subgroup_to_density_[which_processor_group];
   size_t my_group_size = (Ps.size() < 4 ? Ps.size() : my_density_chunk.size());
   return_flag = (int)PFock_setNumDenMat(my_group_size,PFock_);
   if(return_flag != 0)
   {
        outfile->Printf("\n SetP: Ps_size: %d my_group_size: %d", Ps.size(), my_group_size);
        outfile->Printf("\n PFock_setNumDenMat gave an error.");
        throw PSIEXCEPTION("PFock_setNumDenMat is FUBAR");
   }
   if(my_group_size != Ps.size())
   {
       for(auto dens : subgroup_to_density_[which_processor_group])
       {
           printf("\n P%d Dens: %d", GlobalComm_->Me(), dens);
       }
       for(auto proc : subgroup_to_process_[which_processor_group])
       {
           printf("\n P%d Proc: %d", GlobalComm_->Me(), proc);
       }
   }
   double* Buffer;
   for(size_t i=0;i<my_group_size;i++){
      int my_density_index = 0;
      ///If my_group_size is size of Ps, then don't call special shit
      if(my_group_size == Ps.size()) 
      {
        my_density_index = i;
        MyBlock(&Buffer,Ps[i]);
      }
      else {
        my_density_index = my_density_chunk[i];
        MyBlock(&Buffer, Ps[my_density_index]);
      }
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
   for(size_t i=0;i<JorK.size();i++){
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

   int myrank = GlobalComm_->Me();
   int which_processor_group = (myrank % subgroup_ ) + 1;
   std::vector<int> my_density_chunk = subgroup_to_density_[which_processor_group];
   size_t my_group_size = ( Js.size() < 4 ) ? Js.size() : my_density_chunk.size();
   int count = 0;
   ///If J is computed on one subprocess, do below
   if(my_group_size == Js.size())
   {
       GenGetCall(Js,(int)PFOCK_MAT_TYPE_J);
       for(size_t i = 0; i < my_group_size; i++)
        Js[i]->scale(0.5);
   }
   ///Otherwise, you need to pass chunk of Js to GenGetCall
   ///Then Broadcast all that information to Js
   else {
       Timer overall_get_j;
       std::vector<int> my_density_chunk = subgroup_to_density_[which_processor_group];
       std::vector<int> my_process_list  = subgroup_to_process_[which_processor_group];
       std::vector<SharedMatrix> J_subset;
       printf("\n P%d trying to get J", GlobalComm_->Me());
       size_t my_group_size =  my_density_chunk.size();
       int count = 0;
       for(auto my_density : my_density_chunk)
           J_subset.push_back(Js[my_density]);
       Timer gen_get_call;
       GenGetCall(J_subset,(int)PFOCK_MAT_TYPE_J);
       printf("\n P%d call of GenGetCall takes %8.6f s", GlobalComm_->Me(), gen_get_call.get());
       Timer send_j_to_master;
       for(size_t i =0; i < my_group_size; i++)
       {
           int my_density_index = my_density_chunk[i];
           J_subset[i]->scale(0.5);
           ///If on process 0, just direcly modify Js
           if(GlobalComm_->Me() == 0)
           {
               Js[my_density_index] = J_subset[i];
           }
           ///Otherwise, J_subset is contained on list of processors in proc_list
           ///Really only need to send from one of these processors
           ///Take the first one for the hell of it
           else if (GlobalComm_->Me() == my_process_list[0]){
               MPI_Send(&J_subset[i]->pointer()[0][0], NBasis_ * NBasis_, MPI_DOUBLE, 0, my_density_index, MPI_COMM_WORLD);
               printf("\n P%d sending J[%d]", GlobalComm_->Me(), my_density_index);
           }
       }
       printf("\n P%d sending Js to master takes %8.6f s", GlobalComm_->Me(),send_j_to_master.get());

       if(GlobalComm_->Me() == 0)
       {
       Timer get_J;
           for(int group = 2; group < subgroup_ + 1; group++){
               std::vector<int> pzero_density_chunk = subgroup_to_density_[group];
               std::vector<int> pzero_process_list  = subgroup_to_process_[group];
               for(int i = 0; i < my_density_chunk.size(); i++) {
                   int my_true_index = pzero_density_chunk[i];
                   MPI_Recv(&Js[my_true_index]->pointer()[0][0], NBasis_ * NBasis_, MPI_DOUBLE, pzero_process_list[0], my_true_index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               }
               printf("\n group: %d process_list[0]: %d", group, pzero_process_list[0]);
           }
        outfile->Printf("\n Takes %8.6f to recv all J", get_J.get());
       }
       Timer broadcast_j;
       printf("\n P%d about to hit a Bcast", GlobalComm_->Me());
       MPI_Barrier(MPI_COMM_WORLD);
       for(int i = 0; i < Js.size(); i++)
       {
           MPI_Bcast(&Js[i]->pointer()[0][0], NBasis_ * NBasis_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       }
       printf("\n P%d takes %8.6f s to broadcast J", GlobalComm_->Me(), broadcast_j.get());
       printf("\n P%d overall J takes %8.6f s", GlobalComm_->Me(), overall_get_j.get());
   }
}
void psi::MinimalInterface::GetK(std::vector<SharedMatrix>& Ks){
   int myrank = GlobalComm_->Me();
   int which_processor_group = (myrank % subgroup_ ) + 1;
   std::vector<int> my_density_chunk = subgroup_to_density_[which_processor_group];

   size_t my_group_size = (Ks.size() < 4 ? Ks.size() : my_density_chunk.size());
   std::vector<SharedMatrix> K_subset;
   int count = 0;
   
   if(my_group_size == Ks.size())
   {
       GenGetCall(Ks,(int)PFOCK_MAT_TYPE_K);
       for(size_t i = 0; i < my_group_size; i++)
           Ks[i]->scale(-1.0);
   }
   else 
       throw PSIEXCEPTION("Kevin has not implemented subgroup builds for Exchange yet");
       //for(auto my_density : my_density_chunk)
       //    K_subset.push_back(Ks[my_density]);
       //GenGetCall(K_subset, (int)PFOCK_MAT_TYPE_K);
       //for(size_t i =0; i < my_group_size;i++)
       //{
       //     K_subset[i]->scale(-1.0);
       //     int my_density_index = my_density_chunk[i];
       //     Ks[my_density_index] = K_subset[i];
       //     //GlobalComm_->Bcast(&(*(Ks[my_density_index]))(0,0), NBasis_ * NBasis_, 0);
       //     MPI_Bcast(&(Ks[my_density_index]->pointer()[0]), NBasis_ * NBasis_, MPI_DOUBLE, GlobalComm_->Me(), MPI_COMM_WORLD);
       //}
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
   //MPI_Comm_size(density_comm_, &MyRank);
   int MyRank = Comm_->Me();
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
   //int NProc
   //MPI_Comm_rank(density_comm_, &NProc);
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
void psi::MinimalInterface::create_communicators(int NMats, int density_matrices_per_process, int subgroup_number)
{
    int global_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &global_proc);
    if(psi::WorldComm->GetComm()->NProc() != global_proc)
    {
        printf("\n You are trying to recursively create many subcommuncators");
        printf("\n WTF Kevin!!");
        throw PSIEXCEPTION(" Trying to recursively create subcommunicators");

    }

    if(NMats < 4 or global_proc == 1 or (density_matrices_per_process == 0 and subgroup_number == 0))
    {
        ///Create A new communciator that copies the COMM_WORLD
        /// Basically done so same code can be used for sub comm
        std::vector<int> density_list(NMats, 0);
        std::iota(density_list.begin(), density_list.end(), 0);
        subgroup_to_density_[subgroup_] = density_list;
        Comm_ = psi::WorldComm->GetComm()->MakeComm(subgroup_, "WorldComm");
    }
    else {
        if(subgroup_number != 0 && density_matrices_per_process !=0)
        {
              if(NMats != subgroup_number * density_matrices_per_process)
              {
                  outfile->Printf("\n Invalid group numbers and density_matrices_per_process");
                  outfile->Printf("\n You specifed %d subgroup_number and %d density_matrices_per_process", subgroup_number, density_matrices_per_process);
                  throw PSIEXCEPTION("Invalid parameters for Subgroup CASSCF computation");
              }

        }
        else if (subgroup_number == 0 && density_matrices_per_process != 0)
        {
            subgroup_number = NMats / density_matrices_per_process ;
        }
        else if(subgroup_number != 0 && density_matrices_per_process == 0)
        {
            density_matrices_per_process = NMats / subgroup_number ;
        }
        else {
            outfile->Printf("\n You did not specify either subgroup or density");
            outfile->Printf("\n You need to figure out what is going on");
            throw PSIEXCEPTION("Invalid parameters for Subgroup CASSCF computation");
        }
        /// Compute number of processor groups
        subgroup_ = subgroup_number;
        if(NMats % density_matrices_per_process != 0)
        {
            outfile->Printf("\n Kevin has not implemented subgroups for cases when NMats % density != 0");
            outfile->Printf("\n Choose your processors to be a multiple of %d", NMats / density_matrices_per_process);
            throw PSIEXCEPTION("Please choose a multiple of subgroup_ for your processors");
        }
        int total_number_process = psi::WorldComm->GetComm()->NProc();
        int processor_per_group = total_number_process / subgroup_;

        //outfile->Printf("\n subgroup: %d total_number_process: %d processor_per_group: %d", subgroup_, total_number_process, processor_per_group);
        outfile->Printf("\n Running subgroup casscf algorithm with %d subgroups and %d processors per group", subgroup_, processor_per_group);
        int which_processor_group = (psi::WorldComm->GetComm()->Me() % subgroup_);
        std::string Comm_subgroup = "SubDensComm" + std::to_string(which_processor_group);
        Comm_ = psi::WorldComm->GetComm()->MakeComm(which_processor_group, Comm_subgroup);
        for(int i = 1; i < subgroup_ + 1; i++) {
            std::vector<int> density_list;
            if(i != subgroup_)
            {
                density_list.resize(density_matrices_per_process);
                std::iota(density_list.begin(), density_list.end(), (i-1) * density_matrices_per_process);
            }
            else {
                density_list.resize(density_matrices_per_process + NMats % density_matrices_per_process);
                std::iota(density_list.begin(), density_list.end(), (i-1) * density_matrices_per_process);
            }
            subgroup_to_density_[i] = density_list;
        }
        for(int i = 1 ; i < subgroup_ + 1; i++)
        {
            std::vector<int> local_processor_list(processor_per_group, 0);
            int index=0;
            for(int rank = 0; rank < GlobalComm_->NProc(); rank++){
                if(rank % subgroup_ == (i - 1))
                {
                   local_processor_list[index++] = rank;
                }
            }
            subgroup_to_process_[i] = local_processor_list;
        }
    }
}
void psi::MinimalInterface::create_processor_list(std::vector<int>& processor_list, int &processor_size, int total_number_density)
{
   int total_size = GlobalComm_->NProc();
   int total_rank = GlobalComm_->Me();

   if(subgroup_ == 1)
   {
        processor_list.resize(total_size);
        for(int i = 0; i < total_size; i++)
        {
            processor_list[i] = i;
        }
        processor_size = total_size;
   }
   else {
        processor_size = ( total_size % subgroup_ == 0) ? total_size / subgroup_ : -1;
        if(processor_size == -1)
        {
            printf("\n total_number_density: %d subgroup_: %d total_number / subgroup_: %d", total_size, subgroup_,total_size / subgroup_);
            printf("\n total_number_density: %d subgroup_: %d total_number / subgroup_: %d", total_size, subgroup_,total_size / subgroup_);
            throw PSIEXCEPTION("GTFock with subgroups does not work with these numbers");
        }
        int which_processor_group = (total_rank % subgroup_);
        int total_number_process = total_size;
        processor_list = subgroup_to_process_[which_processor_group + 1];

        
    }
}
void MakeBasis(BasisSet** GTBasis, SharedBasis PsiBasis){
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


