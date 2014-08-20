/*
 * libfrag.h
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#ifndef LIBFRAG_H_
#define LIBFRAG_H_

#include <string>
#include <boost/python.hpp>
#include "FragOptions.h"
#include "LibFragTypes.h"
#include "libpsio/MOFile.h"
#include "LibFragBase.h"

namespace psi{
namespace LibFrag {
class GMBE;
class BSSEer;


/** \brief This class is a mess from a c++ point of view; however my python
 *   is too terrible to do this right (mirror the classes comprising this
 *   class in python).
 *
 *   Basically any function that we may need in python from libfrag has
 *   an extension here.
 *
 */
class LibFragHelper : public LibFragBase {
   private:
      /** \brief These are the actual fragments, dimers, etc.
       *
       *   For the MBE this looks like:
       *   Systems[0]={Monomers}
       *   Systems[1]={Dimers}
       *   ...
       *   Systems[n-1]={n-mers}
       *
       *   For the GMBE this looks like:
       *   Systems[0]={Monomers}
       *   Systems[1]={n-mers, positive intersections}
       *   Systems[2]={negative intersections}
       *
       *   Python doesn't need to know what it is running it just
       *   needs to know what to loop over...
       *
       *
       */
      std::vector<NMerSet> Systems;

      ///The actual energy expansion
      boost::shared_ptr<GMBE> Expansion;

      ///The way we are correcting for BSSE (if we are)
      boost::shared_ptr<BSSEer> BSSEFactory;

      ///The factory for adding caps
      boost::shared_ptr<Capper> CapFactory;

      ///The factory for embedding calculations
      static boost::shared_ptr<Embedder> EmbedFactory;

      ///The list of options
      FragOptions DaOptions;

      ///A vector of the MO files
      std::vector<psi::MOFile> MOFiles;

      ///A vector of the charge sets we read in
      std::vector<boost::shared_ptr<double[]> > ChargeSets;

      ///A wrapper for the broadcast/receive operations of Synch
      void BroadCastWrapper(const int i, const int j,std::string& comm,
            std::vector<psi::MOFile>& tempfiles,
            std::vector<boost::shared_ptr<double[]> >& tempChargeSets,
            bool bcast);

   public:
      LibFragHelper();

      ///Sets up the class and makes the fragments
      void Fragment_Helper(boost::python::str& FragMethod, const int N,
            boost::python::str& EmbedMethod,
            boost::python::str& CapMethod,
            boost::python::str& BSSEMethod);

      void NMer_Helper(const int N);

      boost::python::list Embed_Helper(const int N, const int x);

      ///Returns a string: cap_symbol <carts>\n for each cap in "N"-mer "NMer"
      std::string Cap_Helper(const int NMer,const int N);

      ///Returns the highest n-body approximate energy available
      double CalcEnergy(boost::python::list& Energies);

      int GetNNMers(const int i) {
         if (i<Systems.size()) return Systems[i].size();
         else return 0;
      }

      int GetNFrags() {
         return GetNNMers(0);
      }
      boost::python::list GetNMerN(const int NMer, const int N);

      boost::python::list GetGhostNMerN(const int NMer, const int N);

      /** \brief Gathers Relevant data from after each calculation
       *
       *
       *   If we are using the MBE and have not severed any bonds
       *   (laziness on the MBE part, severing bonds makes things complicated)
       *   this will store our MO coefficient files.  Then when we run the
       *   N-mers we call WriteMOs(N,x), where x is the x-th N-Mer, and
       *   we use the direct sum as an initial SCF guess.
       *
       *   This also will read in point charges from wavefunction if we
       *   are using APC embedding
       *
       */
      void GatherData();

      ///Constructs a MOFile that is the direct sum of the N fragments in
      ///the "x"-th "N"-mer
      void WriteMOs(const int N, const int x);

      int IsGMBE();

      ///Are we iterating for whatever embedding reason:
      int Iterate(const int itr);

      ///Turn off self-interaction in point charges
      bool SelfIntOff();

      ///Do we need to run the fragments
      bool RunFrags();

      /** \brief This function is called after a batch is run. It synchronizes
       *         all the processes
       *
       *   \param[in] Comm The communicator we are synching over
       *   \param[in] N The MBE we just completed (determines what data needs
       *                synched)
       */
      void Synchronize(boost::python::str& Comm,const int N,const int itr);
      ~LibFragHelper();
};
}}//End namespaces

#endif /* LIBFRAG_H_ */
