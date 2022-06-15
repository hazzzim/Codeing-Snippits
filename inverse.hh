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

#ifndef IHEATTRANSFER_HH
#define IHEATTRANSFER_HH

#include <cmath>

#include "fem/fixdune.hh"
#include "fem/variables.hh"


template <class RType, class VarSet>
class IHeatFunctional: public FunctionalBase<VariationalFunctional>
{
public:
  using Scalar = RType;
  using OriginVars = VarSet;
  using AnsatzVars = VarSet;
  using TestVars = VarSet;
  using Grid = typename AnsatzVars::Grid;
  using Spaces = typename AnsatzVars::Spaces;
  
  using MeasurementSpace = std::remove_pointer_t<typename boost::fusion::result_of::value_at_c<Spaces,1>::type>;
  using MeasurementData  = typename MeasurementSpace::template Element_t<1>; 
  
  static constexpr int dim = Grid::dimension;

/// \class DomainCache
///

  class DomainCache 
  {
  public:
    DomainCache(IHeatFunctional const& F_,
                typename AnsatzVars::VariableSet const& vars_,
                int flags=7):
      F(F_), data(vars_) 
    {}

    void moveTo(Cell<Grid> const& c) { cell = c; }

    template <class Evaluators>
    void evaluateAt(LocalPosition<Grid> const& x, Evaluators const& evaluators)
    {
      T  = data.template value<0>(evaluators);
      dT = data.template derivative<0>(evaluators)[0];

      lambda  = data.template value<1>(evaluators);
      dlambda = data.template derivative<1>(evaluators)[0];
      
      f  = data.template value<2>(evaluators);
      df = data.template derivative<2>(evaluators)[0];

      
      
      alpha= F.alpha;
      T_m   = F.measurementData.value(boost::fusion::at_c<1>(evaluators));

    }


    Scalar d0() const
    {
      return (T-T_m)*(T-T_m)/2 + (alpha/2 *df*df) + dT*F.kappa*dlambda + T*F.q*lambda - f*lambda ;
    }
    
    template<int row, int dim> 
    Dune::FieldVector<Scalar, TestVars::template Components<row>::m>
    d1 (VariationalArg<Scalar,dim> const& arg) const 
    {
    if(row ==0) return (T-T_m)*arg.value + F.kappa*arg.derivative[0] *dlambda + arg.value*F.q*lambda ;
    if(row ==1) return F.kappa*dT*arg.derivative[0] + F.q*T*arg.value - f*arg.value ; 
    if(row ==2) return alpha* df * arg.derivative[0] -arg.value*lambda ; 

    }

    template<int row, int col, int dim> 
    Dune::FieldMatrix<Scalar, TestVars::template Components<row>::m,
                      AnsatzVars::template Components<col>::m>
    d2 (VariationalArg<Scalar,dim> const &argT,
        VariationalArg<Scalar,dim> const &argA) const
    {
     if(row==0 && col ==0) return argA.value *argT.value;
     if(row==0 && col ==1) return argA.derivative[0]*argT.derivative[0] +  argA.value*argT.value;
     if(row==0 && col ==2) return 0;
     
     if(row==1 && col ==0) return F.kappa*argA.derivative[0]*argT.derivative[0] + F.q *argA.value*argT.value;
     if(row==1 && col ==1) return 0;
     if(row==1 && col ==2) return -argA.value*argT.value;
     
     if(row==2 && col ==0) return 0;
     if(row==2 && col ==1) return -argA.value*argT.value;
     if(row==2 && col ==2) return alpha* argA.derivative[0]*argT.derivative[0]  ;
    }


  private:
    IHeatFunctional const& F;
    typename AnsatzVars::VariableSet const& data;
    Cell<Grid> cell;
    Scalar f;
    Scalar T;
    Scalar lambda;
    Scalar T_m;
    Scalar alpha;
    Dune::FieldVector<Scalar,dim> dT;
    Dune::FieldVector<Scalar,dim> dlambda;
    Dune::FieldVector<Scalar,dim> df;
  };

/// \class BoundaryCache
///

  class BoundaryCache 
  {
  public:
    using FaceIterator = typename AnsatzVars::Grid::LeafIntersectionIterator;

    BoundaryCache(IHeatFunctional const&,
                  typename AnsatzVars::VariableSet const& vars_,
                  int flags=7):
      data(vars_), e(0)
    {}

    void moveTo(FaceIterator const& entity)
    {
      e = &entity;
      penalty = 1.0e14;
    }

    template <class Evaluators>
    void evaluateAt(Dune::FieldVector<typename Grid::ctype,dim-1> const& x, Evaluators const& evaluators)
    {

      T = data.template value<0>(evaluators);
      lambda = data.template value<1>(evaluators);
      f = data.template value<2>(evaluators);
      T0 = 0;   
    }

    Scalar
    d0() const 
    {
      return penalty*(T-T0)*(T-T0)/2 ;
    }
    
    template<int row, int dim> 
    Dune::FieldVector<Scalar, TestVars::template Components<row>::m>
    d1 (VariationalArg<Scalar,dim> const& arg) const 
    {
      if(row==1) return penalty*(T-T0)*arg.value;
      else return 0.0;
    }

    template<int row, int col, int dim> 
    Dune::FieldMatrix<Scalar, TestVars::template Components<row>::m,
                      AnsatzVars::template Components<col>::m>
    d2 (VariationalArg<Scalar,dim> const &arg1, 
                       VariationalArg<Scalar,dim> const &arg2) const 
    {
      if (row==1 && col==0  ) return penalty*arg1.value*arg2.value;
      else return 0.0;
    }

  private:
    typename AnsatzVars::VariableSet const& data;
    FaceIterator const* e;
    Scalar penalty, T, T0, lambda,f;
  };


/// \struct IHeatFunctional constructor
///

 IHeatFunctional(Scalar kappa_, Scalar q_,Scalar alpha_, MeasurementData const& measurementData_): kappa(kappa_), q(q_),alpha(alpha_),measurementData(measurementData_)
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
  Scalar kappa, q,alpha;
  MeasurementData const& measurementData;
};

#endif
