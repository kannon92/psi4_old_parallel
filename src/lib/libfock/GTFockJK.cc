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

#include "jk.h"
#include "../libJKFactory/MinimalInterface.h"
#include <psi4-dec.h>
namespace psi {

///Blank constructor(similar to all other JK)
///KPH wants to hold off on allocating Impl because it needs the number of 
///J and K matrices.

//GTFockJK::GTFockJK(boost::shared_ptr<psi::BasisSet> Primary,
//      size_t NMats,bool AreSymm):
//      JK(Primary),Impl_(new MinimalInterface(Primary, NMats,AreSymm))
//{
//    NMats_ = NMats;
//}
GTFockJK::GTFockJK(boost::shared_ptr<BasisSet> Primary) :
    JK(Primary)
{

}

void GTFockJK::compute_JK() {

   ///KPH: Trying to get GTFock to work with other JK builds
   ///If user did not say how many jk builds are necessary, find this information from jk object.  
   //if(NMats_ == 0)
   //{
   //     outfile->Printf("\n NMats: %d", NMats_);
   //     outfile->Printf("\n Cleft: %d Cright: %d", C_left_.size(), C_right_.size());
   //     NMats_ = C_left_.size(); 
   //     Impl_.reset(new MinimalInterface(NMats_, lr_symmetric_, primary_));
   //     ///Not really sure why this is here
   //     //Need to make sure that NMats is always equal to C_left/C_right
   //     //
   //     NMats_ = 0;
   //}
   //else 
   //{
   //     NMats_ = C_left_.size();
   //}
   NMats_ = C_left_.size();
   MinimalInterface Impl(NMats_, lr_symmetric_);
   Timer SetP_time;
   Impl.SetP(D_ao_);
   printf("\n SetP takes %8.8f s. with %d densities", SetP_time.get(), D_ao_.size());
   Timer GetJK;
   Impl.GetJ(J_ao_);
   Impl.GetK(K_ao_);
   //Impl_->DeleteComm();
   outfile->Printf("\n Get J and K %8.8f s.", GetJK.get());
}
}
