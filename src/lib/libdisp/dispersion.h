/*
 *@BEGIN LICENSE
 *
 * PSI4: an ab initio quantum chemistry software package
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
 *@END LICENSE
 */

#ifndef dispersion_h
#define dispersion_h

/**********************************************************
* dispersion.h: declarations -D(1-3) for KS-DFT
* Robert Parrish, robparrish@gmail.com
* 09/01/2010
*
***********************************************************/
#include <psi4-dec.h>
#include <string>

namespace psi {

class Molecule;
class Matrix;

class Dispersion {

public:
    enum C6_type { C6_arit, C6_geom };
    enum C8_type { C8_geom };
    enum Damping_type { Damping_D1, Damping_CHG, Damping_TT };
    enum Spherical_type { Spherical_Das, Spherical_zero };

protected:

    std::string name_;
    std::string description_;
    std::string citation_;
    std::string bibtex_;

    C6_type C6_type_;
    C8_type C8_type_;
    Damping_type Damping_type_;
    Spherical_type Spherical_type_;

    double s6_;
    double d_;
    double sr6_;
    double s8_;
    double a1_;
    double a2_;
    const double *RvdW_;
    const double *C6_;
    const double *C8_;
    const double *A_;
    const double *Beta_;

public:

    Dispersion();
    virtual ~Dispersion();

    static boost::shared_ptr<Dispersion> build(const std::string & type, double s6 = 0.0, 
        double p1 = 0.0, double p2 = 0.0, double p3 = 0.0);

    std::string name() const { return name_; }
    std::string description() const { return description_; }
    std::string citation() const { return citation_; }
    std::string bibtex() const { return bibtex_; }
    void set_name(const std::string & name) { name_ = name; }
    void set_description(const std::string & description) { description_ = description; }
    void set_citation(const std::string & citation) { citation_ = citation; }
    void set_bibtex(const std::string & bibtex) { bibtex_ = bibtex; }

    boost::shared_ptr<Vector> set_atom_list(boost::shared_ptr<Molecule> mol);

    double get_d() const { return d_; }
    double get_s6() const { return s6_; }
    double get_sr6() const { return sr6_; }
    double get_s8() const { return s8_; }
    double get_a1() const { return a1_; }
    double get_a2() const { return a2_; }

    void set_d(double d) { d_ = d; }
    void set_s6(double s6) { s6_ = s6; }
    void set_sr6(double sr6) { sr6_ = sr6; }
    void set_s8(double s8) { s8_ = s8; }
    void set_a1(double a1) { a1_ = a1; }
    void set_a2(double a2) { a2_ = a2; }

    std::string print_energy(boost::shared_ptr<Molecule> m);
    std::string print_gradient(boost::shared_ptr<Molecule> m);
    std::string print_hessian(boost::shared_ptr<Molecule> m);

    virtual double compute_energy(boost::shared_ptr<Molecule> m);
    virtual SharedMatrix compute_gradient(boost::shared_ptr<Molecule> m);
    virtual SharedMatrix compute_hessian(boost::shared_ptr<Molecule> m);

    virtual void print(FILE* out = outfile, int level = 1) const;
    void py_print() const { print(outfile, 1); }
};

}

#endif
