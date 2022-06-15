/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*  This file is part of the library KASKADE 7                               */
/*    see http://www.zib.de/projects/kaskade7-finite-element-toolbox         */
/*                                                                           */
/*  Copyright (C) 2002-2022 Zuse Institute Berlin                            */
/*                                                                           */
/*  KASKADE 7 is distributed under the terms of the ZIB Academic License.    */
/*    see $KASKADE/academic.txt                                              */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HEATTRANSFER_HH
#define HEATTRANSFER_HH

#include <cmath>

#include "fem/fixdune.hh"
#include "fem/variables.hh"

/// Example stationary heat transfer equation
///

template <class RType, class VarSet>
class HeatFunctional: public FunctionalBase<VariationalFunctional>
{
public:
  using Scalar = RType;
  using OriginVars = VarSet;
  using AnsatzVars = VarSet;
  using TestVars = VarSet;
  using Grid = typename AnsatzVars::Grid;
  static constexpr int dim = Grid::dimension;

/// \class DomainCache
///

  class DomainCache 
  {
  public:
    DomainCache(HeatFunctional const& F_,
                typename AnsatzVars::VariableSet const& vars_,
                int flags=7):
      F(F_), data(vars_) 
    {}

    void moveTo(Cell<Grid> const& c) { cell = c; }

    template <class Evaluators>
    void evaluateAt(LocalPosition<Grid> const& x, Evaluators const& evaluators)
    {
      u  = data.template value<0>(evaluators);
      du = data.template derivative<0>(evaluators)[0];
      
      xglob = cell.geometry().global(x);

      if (xglob[0] >0.25 && xglob[0]<0.75 && xglob[1] >0.25 && xglob[1]<0.75 ){
       f=1e5;
      }
      else f=0.0;
    }

    Scalar d0() const
    {
      return (F.kappa*du*du + F.q*u*u)/2 - f*u;
    }
    
    template<int row, int dim> 
    Dune::FieldVector<Scalar, TestVars::template Components<row>::m>
    d1 (VariationalArg<Scalar,dim> const& arg) const 
    {
      return F.kappa*du*arg.derivative[0] + F.q*u*arg.value - f*arg.value; 
    }

    template<int row, int col, int dim> 
    Dune::FieldMatrix<Scalar, TestVars::template Components<row>::m,
                      AnsatzVars::template Components<col>::m>
    d2 (VariationalArg<Scalar,dim> const &argT,
        VariationalArg<Scalar,dim> const &argA) const
    {
      return F.kappa*argT.derivative[0]*argA.derivative[0] + F.q*argT.value*argA.value;
    }

  private:
    HeatFunctional const& F;
    typename AnsatzVars::VariableSet const& data;
    Cell<Grid> cell;
    GlobalPosition<Grid> xglob;
    Scalar f;
    Scalar u;
    Dune::FieldVector<Scalar,dim> du;
  };

/// \class BoundaryCache
///

  class BoundaryCache 
  {
  public:
    using FaceIterator = typename AnsatzVars::Grid::LeafIntersectionIterator;

    BoundaryCache(HeatFunctional const&,
                  typename AnsatzVars::VariableSet const& vars_,
                  int flags=7):
      data(vars_), e(0)
    {}

    void moveTo(FaceIterator const& entity)
    {
      e = &entity;
      penalty = 1.0e9;
    }

    template <class Evaluators>
    void evaluateAt(Dune::FieldVector<typename Grid::ctype,dim-1> const& x, Evaluators const& evaluators)
    {
      xglob = (*e)->geometry().global(x);

      u = data.template value<0>(evaluators);
      u0 = 0;   
    }

    Scalar
    d0() const 
    {
      return penalty*(u-u0)*(u-u0)/2;
    }
    
    template<int row, int dim> 
    Dune::FieldVector<Scalar, TestVars::template Components<row>::m>
    d1 (VariationalArg<Scalar,dim> const& arg) const 
    {

        return penalty*(u-u0)*arg.value;

    }

    template<int row, int col, int dim> 
    Dune::FieldMatrix<Scalar, TestVars::template Components<row>::m,
                      AnsatzVars::template Components<col>::m>
    d2 (VariationalArg<Scalar,dim> const &arg1, 
                       VariationalArg<Scalar,dim> const &arg2) const 
    {

        return penalty*arg1.value*arg2.value;

    }

  private:
    typename AnsatzVars::VariableSet const& data;
    FaceIterator const* e;
    GlobalPosition<Grid> xglob;
    Scalar penalty, u, u0;
  };


/// \struct HeatFunctional constructor
///

  HeatFunctional(Scalar kappa_, Scalar q_): kappa(kappa_), q(q_)
  {
  }
  
  
  
/// \struct D2
///

  template <int row>
  struct D1: public FunctionalBase<VariationalFunctional>::D1<row> 
  {
    static bool const present   = true;
    static bool const constant  = false;
  };
   
public:
  template <int row, int col>
  struct D2: public FunctionalBase<VariationalFunctional>::D2<row,col>   
  {
    static bool const present   = true;
    static bool const symmetric = true;
    static bool const lumped    = false;
  };

/// \fn integrationOrder
///

  template <class Cell>
  int integrationOrder(Cell const& /* cell */,
                       int shapeFunctionOrder, bool boundary) const 
  {
      return 2*shapeFunctionOrder;
  }
  
private:
  Scalar kappa, q;
};

/// \example ht.cpp
/// show the usage of HeatFunctional describing a stationary heat transfer problem,
/// no adaptive grid refinement.
///
#endif
