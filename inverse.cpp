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

#include <iostream>
#include <string>                     
#include <random>

#include <boost/timer/timer.hpp>

#include "dune/grid/config.h"
#include "dune/grid/uggrid.hh"

#include "fem/assemble.hh"
#include "fem/norms.hh"
#include "fem/lagrangespace.hh"
#include "linalg/direct.hh"
#include "utilities/enums.hh"
#include "utilities/gridGeneration.hh" 
#include "io/vtk.hh"
#include "utilities/kaskopt.hh"

using namespace Kaskade;
#include "ht.hh"
#include "inverse.hh"


int main(int argc, char *argv[])
{
  
  using namespace boost::fusion;
  


  std::cout << "Start inverse heat transfer tutorial program " << std::endl;

  boost::timer::cpu_timer totalTimer;

  int verbosityOpt = 1;
  bool dump = false;


  std::unique_ptr<boost::property_tree::ptree> pt = getKaskadeOptions(argc, argv, verbosityOpt, dump);

  int  refinements = getParameter(pt, "refinements", 5),
       order       = getParameter(pt, "order", 1);
       
  double   kappa   = getParameter(pt, "kappa", 1.0),
           q       = getParameter(pt, "q", 1.0),
	   alpha   = getParameter(pt,"alpha",1e-6),
	   noise   = getParameter(pt,"noise",1);
       
       
  std::cout << "original mesh shall be refined           : " << refinements << " times" << std::endl;
  std::cout << "discretization order                     : " << order << std::endl;
  std::cout << "the diffusion constant kappa is          : " << kappa << std::endl;
  std::cout << "the mass coefficient q is                : " << q << std::endl;
  std::cout << "standard deviation of the Gaussian noise : " << noise << std::endl;


  constexpr int dim=2;        
  using Grid = Dune::UGGrid<dim>;
  using LeafView = Grid::LeafGridView;
  using H1Space  = FEFunctionSpace<ContinuousLagrangeMapper<double,LeafView> >;
  using L2Space  =FEFunctionSpace<DiscontinuousLagrangeMapper<double,LeafView>>;
  using Spaces   = boost::fusion::vector<H1Space const*, L2Space const*>;
  using VariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<1>,VariableId<0> > >;
  using VariableSet = VariableSetDescription<Spaces,VariableDescriptions>;
  using Functional = HeatFunctional<double,VariableSet>;
  using Assembler = VariationalFunctionalAssembler<LinearizationAt<Functional> >;





  // Gemometry creation and Mesh refinement  
  GridManager<Grid> gridManager( createUnitSquare<Grid>() );
  gridManager.globalRefine(refinements);

  
  // construction of finite element space for the scalar solution T_m.
  H1Space h1Space(gridManager,gridManager.grid().leafGridView(),order);

  //construction of the L2 space for measurement storage
  L2Space l2Space(gridManager,gridManager.grid().leafGridView(),0);

  Spaces spaces(&h1Space,&l2Space);
  
  //construction of L2 space with 1 component for storage
  L2Space::Element_t<1> measurementSpace(l2Space);
  L2Space::Element_t<1> sourceSpace(l2Space);

  
  // start of the forward problem set up

  // construct variable list.
        
  std::string varNames[1] = { "T_m" };
  std::string sourceVarNames[1] = { "source" };

    
  VariableSet variableSet(spaces,varNames);
  VariableSet sourceVariableSet(spaces,sourceVarNames);


  // construct variational functional
    
  Functional F(kappa,q);
  

  constexpr int neq = Functional::TestVars::noOfVariables;
  constexpr int nvars = Functional::AnsatzVars::noOfVariables;
  size_t dofs = variableSet.degreesOfFreedom(0,nvars);
  std::cout << std::endl << "no of variables = " << nvars << std::endl;
  std::cout << "no of equations = " << neq   << std::endl;
  std::cout << "number of degrees of freedom = " << dofs   << std::endl;

  //construct Galerkin representation
  Assembler assembler(spaces);
  VariableSet::VariableSet T_m(variableSet);
  VariableSet::VariableSet source(sourceVariableSet);

  
  boost::timer::cpu_timer assembTimer;
  assembler.assemble(linearization(F,T_m));
  std::cout << "computing time for assemble: " << boost::timer::format(assembTimer.elapsed()) << "\n";
  
  auto  rhs      = assembler.rhs();
  auto  solution = variableSet.zeroCoefficientVector();
	
    AssembledGalerkinOperator<Assembler> A(assembler);
  

   //Using a direct solver
   
    boost::timer::cpu_timer directTimer;
    directInverseOperator(A,DirectType::UMFPACK,MatrixProperties::GENERAL).applyscaleadd(-1.0,rhs,solution);
    T_m.data = solution.data;
    std::cout << "computing time for direct solve: " << boost::timer::format(directTimer.elapsed()) << "\n";  




  // Data generation: Storing and adding noise to the generated measurement field by using interpolate globally

  // create a random number generator 
  std::default_random_engine generator;


  interpolateGlobally<PlainAverage>(measurementSpace,makeFunctionView(h1Space, [&] (auto const& evaluator)
  {

    double measurement= component<0>(T_m).value(evaluator) ;
    std::normal_distribution<double> distribution(measurement,noise);


    return Dune::FieldVector<double,1>( distribution(generator));
  
  }));

  //Storing the source field for visualization

  interpolateGlobally<PlainAverage>(sourceSpace,makeFunctionView(h1Space, [&] (auto const& evaluator)
  {

    auto   x = evaluator.cell().geometry().center();
    double f=0;
   
   if (x[0] >0.25 && x[0]<0.75 && x[1] >0.25 && x[1]<0.75 ) f=1e5;

    return Dune::FieldVector<double,1>(f);
  
  }));


  
 //start of the inverse problem set up
  using IVariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<1>,VariableId<0>>,
                                                      Variable<SpaceIndex<0>,Components<1>,VariableId<1>>,
						      Variable<SpaceIndex<0>,Components<1>,VariableId<2>>>;

  using IVariableSet        = VariableSetDescription<Spaces,IVariableDescriptions>;
  using IFunctional         = IHeatFunctional<double,IVariableSet>;
  using IAssembler          = VariationalFunctionalAssembler<LinearizationAt<IFunctional> >;

   
 
  std::string IvarNames[3] = { "T","lambda","f" };
  IVariableSet IvariableSet(spaces,IvarNames);
  IFunctional  IF(kappa,q,alpha,measurementSpace);
  
  
  
  constexpr int Invars = IFunctional::AnsatzVars::noOfVariables;
  constexpr int Ineq   = IFunctional::TestVars::noOfVariables;
  size_t Idofs         = IvariableSet.degreesOfFreedom(0,Invars);
  std::cout << std::endl << "no of variables of Inverse system = " << Invars << std::endl;
  std::cout << "no of equations of Inverse system = " << Ineq   << std::endl;
  std::cout << "number of degrees of freedom of Inverse system = " << Idofs   << std::endl;
  
  IAssembler Iassembler(spaces);
  IVariableSet::VariableSet inverseVariables(IvariableSet);
  
  boost::timer::cpu_timer IassembTimer;
  Iassembler.assemble(linearization(IF,inverseVariables));
  std::cout << "computing time for inverse assemble: " << boost::timer::format(IassembTimer.elapsed()) << "\n";
  

  auto  Irhs      = Iassembler.rhs();
  auto  Isolution = IvariableSet.zeroCoefficientVector();
  
  boost::timer::cpu_timer IdirectTimer;
  
  AssembledGalerkinOperator<IAssembler> IA(Iassembler);
  

  directInverseOperator(IA,DirectType::UMFPACK,MatrixProperties::GENERAL).applyscaleadd(-1.0,Irhs,Isolution);
  inverseVariables.data = Isolution.data;
  std::cout << "computing time for inverse direct solve: " << boost::timer::format(IdirectTimer.elapsed()) << "\n";  
   

  // output of solution in VTK format.

  boost::timer::cpu_timer outputTimer;
  component<0>(source)= sourceSpace;
  writeVTKFile(inverseVariables,"temperature_inverse",IoOptions().setOrder(order).setPrecision(7));
  writeVTKFile(T_m,"temperature",IoOptions().setOrder(order).setPrecision(7));
  writeVTKFile(source,"Source",IoOptions().setOrder(order).setPrecision(7));

  std::cout << "graphical output finished, data in VTK format is written into file Source_term" << ".vtu \n";

  std::cout << "computing time for output: " << boost::timer::format(outputTimer.elapsed()) << "\n";

  std::cout << "total computing time: " << boost::timer::format(totalTimer.elapsed()) << "\n";
  std::cout << "End Inverse tutorial program" << std::endl;


}
