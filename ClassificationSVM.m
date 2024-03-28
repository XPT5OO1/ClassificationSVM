## Copyright (C) 2024 Pallav Purbia <pallavpurbia@gmail.com>
## Copyright (C) 2024 Andreas Bertsatos <abertsatos@biol.uoa.gr>
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

classdef ClassificationSVM

## -*- texinfo -*-
## @deftypefn  {statistics} {@var{obj} =} ClassificationSVM (@var{X}, @var{Y})
## @deftypefnx {statistics} {@var{obj} =} ClassificationSVM (@dots{}, @var{name}, @var{value})
##
## Create a @qcode{ClassificationSVM} class object containing a Support Vector
## Machine (SVM) for classification.
##
## A @qcode{ClassificationSVM} class object can store the predictors and response
## data along with various parameters for the SVM model.  It is recommended to
## use the @code{fitcsvm} function to create a @qcode{ClassificationSVM} object.
##
## @code{@var{obj} = ClassificationSVM (@var{X}, @var{Y})} returns an object of
## class ClassificationSVM, with matrix @var{X} containing the predictor data and
## vector @var{Y} containing the response data.
##
## @itemize
## @item
## @var{X} must be a @math{NxP} numeric matrix of input data where rows
## correspond to observations and columns correspond to features or variables.
## @var{X} will be used to train the SVM model.
## @item
## @var{Y} must be @math{Nx1} numeric vector containing the response data
## corresponding to the predictor data in @var{X}. @var{Y} must have same
## number of rows as @var{X}.
## @end itemize
##
## @code{@var{obj} = ClassificationSVM (@dots{}, @var{name}, @var{value})} returns
## an object of class ClassificationSVM with additional properties specified by
## @qcode{Name-Value} pair arguments listed below.
##
## @multitable @columnfractions 0.05 0.2 0.75
## @headitem @tab @var{Name} @tab @var{Value}
##
## @item @tab @qcode{"Alpha"} @tab Predictor Variable names, specified as
## a row vector cell of strings with the same length as the columns in @var{X}.
## If omitted, the program will generate default variable names
## @qcode{(x1, x2, ..., xn)} for each column in @var{X}.
##
## @item @tab @qcode{"BoxConstraint"} @tab
##
## @item @tab @qcode{"CacheSize"} @tab
##
## @item @tab @qcode{"CategoricalPredictors"} @tab
##
## @item @tab @qcode{"ClassNames"} @tab
##
## @item @tab @qcode{"ClipAlphas"} @tab
##
## @item @tab @qcode{"Cost"} @tab
##
## @item @tab @qcode{"CrossVal"} @tab
##
## @item @tab @qcode{"CVPartition"} @tab
##
## @item @tab @qcode{"Holdout"} @tab
##
## @item @tab @qcode{"KFold"} @tab
##
## @item @tab @qcode{"Leaveout"} @tab
##
## @item @tab @qcode{"GapTolerance"} @tab
##
## @item @tab @qcode{"DeltaGradientTolerance"} @tab
##
## @item @tab @qcode{"KKTTolerance"} @tab
##
## @item @tab @qcode{"IterationLimit"} @tab
##
## @item @tab @qcode{"KernelFunction"} @tab Supported Kernel Functions are
## linear, gaussian (or rbf), polynomial
##
## @item @tab @qcode{"KernelScale"} @tab
##
## @item @tab @qcode{"KernelOffset"} @tab
##
## @item @tab @qcode{"OptimizeHyperparameters"} @tab
##
## @item @tab @qcode{"PolynomialOrder"} @tab
##
## @item @tab @qcode{"Nu"} @tab
##
## @item @tab @qcode{"NumPrint"} @tab
##
## @item @tab @qcode{"OutlierFraction"} @tab
##
## @item @tab @qcode{"PredictorNames"} @tab
##
## @item @tab @qcode{"Prior"} @tab
##
## @item @tab @qcode{"RemoveDuplicates"} @tab
##
## @item @tab @qcode{"ResponseName"} @tab
##
## @item @tab @qcode{"ScoreTransform"} @tab
##
## @item @tab @qcode{"Solver"} @tab
##
## @item @tab @qcode{"ShrinkagePeriod"} @tab
##
## @item @tab @qcode{"Standardize"} @tab
##
## @item @tab @qcode{"Verbose"} @tab
##
## @item @tab @qcode{"Weights"} @tab
##
## @item @tab @qcode{"DeltaGradientTolerance"} @tab
##
## @end multitable
##
## @seealso{fitcsvm}
## @end deftypefn

  properties (Access = public)

    X = [];                   # Predictor data
    Y = [];                   # Class labels

    NumObservations = [];     # Number of observations in training dataset
    RowsUsed        = [];     # Rows used in fitting
    Standardize     = [];     # Flag to standardize predictors
    Sigma           = [];     # Predictor standard deviations
    Mu              = [];     # Predictor means

    NumPredictors   = [];     # Number of predictors
    PredictorNames  = [];     # Predictor variables names
    ResponseName    = [];     # Response variable name
    ClassNames      = [];     # Names of classes in Y
    BreakTies       = [];     # Tie-breaking algorithm
    Prior           = [];     # Prior probability for each class
    Cost            = [];     # Cost of misclassification

    NumNeighbors    = [];     # Number of nearest neighbors
    Distance        = [];     # Distance metric
    DistanceWeight  = [];     # Distance weighting function
    DistParameter   = [];     # Parameter for distance metric
    NSMethod        = [];     # Nearest neighbor search method
    IncludeTies     = [];     # Flag for handling ties
    BucketSize      = [];     # Maximum data points in each node
  ## Supported kernel functions: "gaussian", "rbf", "linear", "polynomial"
  endproperties


  methods (Access = public)

    ## Class object constructor
    function this = ClassificationSVM (X, Y, varargin)
      ## Check for sufficient number of input arguments
      if (nargin < 2)
        error ("ClassificationSVM: too few input arguments.");
      endif

      ## Get training sample size and number of variables in training data
      nsample = rows (X);
      ndims_X = columns (X);

      ## Check correspodence between predictors and response
      if (nsample != rows (Y))
        error ("ClassificationSVM: number of rows in X and Y must be equal.");
      endif

      ## Set default values before parsing optional parameters
      Alpha                   = ;
      BoxConstraint           = 1;
      CacheSize               = 1000;
      CategoricalPredictors   = ;
      ClassNames              = ;
      ClipAlphas              = ;
      Cost                    = ;
      CrossVal                = 'off';
      CVPartition             = ;
      Holdout                 = [];
      KFold                   = 10;
      Leaveout                = ;
      GapTolerance            = 0;
      DeltaGradientTolerance  = [];
      KKTTolerance            = [];
      IterationLimit          = 1e6;
      KernelFunction          = 'linear';
      KernelScale             = 1;
      KernelOffset            = ;
      OptimizeHyperparameters = ;
      PolynomialOrder         = 3;
      Nu                      = ;
      NumPrint                = ;
      OutlierFraction         = ;
      PredictorNames          = {};           #Predictor variable names
      Prior                   = ;
      RemoveDuplicates        = ;
      ResponseName            = [];           #Response variable name
      ScoreTransform          = ;
      Solver                  = ;
      ShrinkagePeriod         = ;
      Standardize             = ;
      Verbose                 = ;
      Weights                 = ;

      ## Number of parameters for Knots, DoF, Order (maximum 2 allowed)
      KOD = 0;
      ## Number of parameters for Formula, Ineractions (maximum 1 allowed)
      F_I = 0;

      ## Parse extra parameters
      while (numel (varargin) > 0)
        switch (tolower (varargin {1}))

          case "alpha"

          case "boxconstraint"

          case "cachesize"

          case "categoricalpredictors"

          case "classnames"

          case "clipalphas"

          case "cost"
            Cost = varargin{2};
            if (! (isnumeric (Cost) && issquare (Cost)))
              error ("ClassificationSVM: Cost must be a numeric square matrix.");
            endif

          case "crossval"

          case "cvpartition"

          case "holdout"
            Holdout = varargin{2};
            if (! isnumeric(Holdout) && isscalar(Holdout))
              error ("ClassificationSVM: Holdout must be a numeric scalar.");
            endif
            if (Holdout < 0 || Holdout >1)
              error ("ClassificationSVM: Holdout must be between 0 and 1.");
            endif


          case "kfold"
            KFold = varargin{2};
            if (! isnumeric(KFold))
              error ("ClassificationSVM: KFold must be a numeric value.");
            endif

          case "leaveout"

          case "gaptolerance"
            GapTolerance = varargin{2};
            if (! isnumeric(GapTolerance) && isscalar(GapTolerance))
              error ("ClassificationSVM: GapTolerance must be a numeric scalar.");
            endif
            if (GapTolerance < 0)
              error ("ClassificationSVM: GapTolerance must be non-negative scalar.");
            endif

          case "deltagradienttolerance"
            DeltaGradientTolerance = varargin{2};
            if (! isnumeric(DeltaGradientTolerance))
              error (strcat(["ClassificationSVM: DeltaGradientTolerance must ], ...
              ["be a numeric value."]));
            endif
            if (GapTolerance < 0)
              error (strcat(["ClassificationSVM: DeltaGradientTolerance must ], ...
              ["be non-negative scalar."]));
            endif


          case "kkttolerance"

          case "iterationlimit"
            IterationLimit = varargin{2};
            if (! isnumeric(IterationLimit) && isscalar(IterationLimit)...
              && IterationLimit >= 0)
              error ("ClassificationSVM: IterationLimit must be a positive number.");
            endif

          case "kernelfunction"
            kernelfunction = varargin{2};
            if (! any (strcmpi (kernelfunction, {"linear", "gaussian", "rbf", ...
              "polynomial"})))
            error ("ClassificationSVM: unsupported Kernel function.");
            endif

          case "kernenlscale"

          case "kerneloffset"
            KernelOffset = varargin{2};
            if (! isnumeric(KernelOffset) && isscalar(KernelOffset)...
              && KernelOffset >= 0)
              error ("ClassificationSVM: KernelOffset must be a non-negative scalar.");
            endif

          case "optimizehyperparameters"

          case "polynomialorder"
            PolynomialOrder = varargin{2};
            if (! isnumeric(PolynomialOrder))
              error ("ClassificationSVM: PolynomialOrder must be a numeric value.");
            endif

          case "nu"

          case "numprint"

          case "outlierfraction"

          case "predictornames"
            PredictorNames = varargin{2};
            if (! isempty (PredictorNames))
              if (! iscellstr (PredictorNames))
                error (strcat (["ClassificationSVM: PredictorNames must"], ...
                               [" be a cellstring array."]));
              elseif (columns (PredictorNames) != columns (X))
                error (strcat (["ClassificationSVM: PredictorNames must"], ...
                               [" have same number of columns as X."]));
              endif
            endif

          case "prior"

          case "removeduplicates"

          case "responsename"
            if (! ischar (varargin{2}))
              error ("ClassificationSVM: ResponseName must be a char string.");
            endif
            ResponseName = tolower(varargin{2});

          case "scoretransform"

          case "solver"
            if (! ischar (varargin{2}))
              error ("ClassificationSVM: Solver must be a char string.");
            endif
            Solver = tolower(varargin{2});
            if(Solver=='smo')
              if(isempty(KernelOffset))
                KernelOffset = 0;
              elseif(isempty(DeltaGradientTolerance))
                DeltaGradientTolerance = 1e-3;
              elseif(isempty(KKTTolerance))
                KKTTolerance = 0;
            elseif(Solver=='isda')
              if(isempty(KernelOffset))
                KernelOffset = 0.1;
              elseif(isempty(DeltaGradientTolerance))
                DeltaGradientTolerance = 0;
              elseif(isempty(KKTTolerance))
                KKTTolerance = 1e-3;
            endif

          case "shrinkageperiod"

          case "standardize"

          case "verbose"

          case "weights"


          case "formula"
            if (F_I < 1)
              Formula = varargin{2};
              if (! ischar (Formula) && ! islogical (Formula))
                error ("ClassificationSVM: Formula must be a string.");
              endif
              F_I += 1;
            else
              error ("ClassificationSVM: Interactions have been already defined.");
            endif

          case "interactions"
            if (F_I < 1)
              tmp = varargin{2};
              if (isnumeric (tmp) && isscalar (tmp)
                                  && tmp == fix (tmp) && tmp >= 0)
                Interactions = tmp;
              elseif (islogical (tmp))
                Interactions = tmp;
              elseif (ischar (tmp) && strcmpi (tmp, "all"))
                Interactions = tmp;
              else
                error ("ClassificationSVM: invalid Interactions parameter.");
              endif
              F_I += 1;
            else
              error ("ClassificationSVM: Formula has been already defined.");
            endif

          case "knots"
            if (KOD < 2)
              Knots = varargin{2};
              if (! isnumeric (Knots) || ! (isscalar (Knots) ||
                  isequal (size (Knots), [1, ndims_X])))
                error ("ClassificationSVM: invalid value for Knots.");
              endif
              DoF = Knots + Order;
              Order = DoF - Knots;
              KOD += 1;
            else
              error ("ClassificationSVM: DoF and Order have been set already.");
            endif

          case "order"
            if (KOD < 2)
              Order = varargin{2};
              if (! isnumeric (Order) || ! (isscalar (Order) ||
                  isequal (size (Order), [1, ndims_X])))
                error ("ClassificationSVM: invalid value for Order.");
              endif
              DoF = Knots + Order;
              Knots = DoF - Order;
              KOD += 1;
            else
              error ("ClassificationSVM: DoF and Knots have been set already.");
            endif

          case "dof"
            if (KOD < 2)
              DoF = varargin{2};
              if (! isnumeric (DoF) ||
                  ! (isscalar (DoF) || isequal (size (DoF), [1, ndims_X])))
                error ("ClassificationSVM: invalid value for DoF.");
              endif
              Knots = DoF - Order;
              Order = DoF - Knots;
              KOD += 1;
            else
              error ("ClassificationSVM: Knots and Order have been set already.");
            endif

          case "tol"
            Tol = varargin{2};
            if (! (isnumeric (Tol) && isscalar (Tol) && (Tol > 0)))
              error ("ClassificationSVM: Tolerance must be a Positive scalar.");
            endif

          otherwise
            error (strcat (["ClassificationSVM: invalid parameter name"],...
                           [" in optional pair arguments."]));

        endswitch
        varargin (1:2) = [];
      endwhile
    endfunction
   endmethods
endclassdef
%!demo
%! ## Train a ClassificationSVM Model for synthetic values
%! f1 = @(x) cos (3 * x);
%! f2 = @(x) x .^ 3;
%! x1 = 2 * rand (50, 1) - 1;
%! x2 = 2 * rand (50, 1) - 1;
%! y = f1(x1) + f2(x2);
%! y = y + y .* 0.2 .* rand (50,1);
%! X = [x1, x2];
%! a = fitrgam (X, y, "tol", 1e-3)

%!demo
%! ## Declare two different functions
%! f1 = @(x) cos (3 * x);
%! f2 = @(x) x .^ 3;
%!
%! ## Generate 80 samples for f1 and f2
%! x = [-4*pi:0.1*pi:4*pi-0.1*pi]';
%! X1 = f1 (x);
%! X2 = f2 (x);
%!
%! ## Create a synthetic response by adding noise
%! rand ("seed", 3);
%! Ytrue = X1 + X2;
%! Y = Ytrue + Ytrue .* 0.2 .* rand (80,1);
%!
%! ## Assemble predictor data
%! X = [X1, X2];
%!
%! ## Train the GAM and test on the same data
%! a = fitrgam (X, Y, "order", [5, 5]);
%! [ypred, ySDsd, yInt] = predict (a, X);
%!
%! ## Plot the results
%! figure
%! [sortedY, indY] = sort (Ytrue);
%! plot (sortedY, "r-");
%! xlim ([0, 80]);
%! hold on
%! plot (ypred(indY), "g+")
%! plot (yInt(indY,1), "k:")
%! plot (yInt(indY,2), "k:")
%! xlabel ("Predictor samples");
%! ylabel ("Response");
%! title ("actual vs predicted values for function f1(x) = cos (3x) ");
%! legend ({"Theoretical Response", "Predicted Response", "Prediction Intervals"});
%!
%! ## Use 30% Holdout partitioning for training and testing data
%! C = cvpartition (80, "Holdout", 0.3);
%! [ypred, ySDsd, yInt] = predict (a, X(test(C),:));
%!
%! ## Plot the results
%! figure
%! [sortedY, indY] = sort (Ytrue(test(C)));
%! plot (sortedY, 'r-');
%! xlim ([0, sum(test(C))]);
%! hold on
%! plot (ypred(indY), "g+")
%! plot (yInt(indY,1),'k:')
%! plot (yInt(indY,2),'k:')
%! xlabel ("Predictor samples");
%! ylabel ("Response");
%! title ("actual vs predicted values for function f1(x) = cos (3x) ");
%! legend ({"Theoretical Response", "Predicted Response", "Prediction Intervals"});

## Test constructor
%!test
%! x = [1, 2, 3; 4, 5, 6; 7, 8, 9; 3, 2, 1];
%! y = [1; 2; 3; 4];
%! a = ClassificationSVM (x, y);
%! assert ({a.X, a.Y}, {x, y})
%! assert ({a.BaseModel.Intercept}, {2.5000})
%! assert ({a.Knots, a.Order, a.DoF}, {[5, 5, 5], [3, 3, 3], [8, 8, 8]})
%! assert ({a.NumObservations, a.NumPredictors}, {4, 3})
%! assert ({a.ResponseName, a.PredictorNames}, {"Y", {"x1", "x2", "x3"}})
%! assert ({a.Formula}, {[]})
%!test
%! x = [1, 2, 3, 4; 4, 5, 6, 7; 7, 8, 9, 1; 3, 2, 1, 2];
%! y = [1; 2; 3; 4];
%! pnames = {"A", "B", "C", "D"};
%! formula = "Y ~ A + B + C + D + A:C";
%! intMat = logical ([1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1;1,0,1,0]);
%! a = ClassificationSVM (x, y, "predictors", pnames, "formula", formula);
%! assert ({a.IntMatrix}, {intMat})
%! assert ({a.ResponseName, a.PredictorNames}, {"Y", pnames})
%! assert ({a.Formula}, {formula})

## Test input validation for constructor
%!error<ClassificationSVM: too few input arguments.> ClassificationSVM ()
%!error<ClassificationSVM: too few input arguments.> ClassificationSVM (ones(10,2))
%!error<ClassificationSVM: number of rows in X and Y must be equal.> ...
%! ClassificationSVM (ones(10,2), ones (5,1))
%!error<ClassificationSVM: unsupported Kernel function.>
%! ClassificationSVM (ones(10,2), ones (10,1), "KernelFunction","some")
%!error<ClassificationSVM: PredictorNames must be a cellstring array.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "PredictorNames", -1)
%!error<ClassificationSVM: PredictorNames must be a cellstring array.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "PredictorNames", ['a','b','c'])
%!error<ClassificationSVM: PredictorNames must have same number of columns as X.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "PredictorNames", {'a','b','c'})

%!error<ClassificationSVM: invalid values in X.> ...
%! ClassificationSVM ([1;2;3;"a";4], ones (5,1))
%!error<ClassificationSVM: invalid parameter name in optional pair arguments.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "some", "some")
%!error<ClassificationSVM: Formula must be a string.>
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", {"y~x1+x2"})
%!error<ClassificationSVM: Formula must be a string.>
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", [0, 1, 0])
%!error<ClassificationSVM: invalid syntax in Formula.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", "something")
%!error<ClassificationSVM: no predictor terms in Formula.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", "something~")
%!error<ClassificationSVM: no predictor terms in Formula.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", "something~")
%!error<ClassificationSVM: some predictors have not been identified> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", "something~x1:")
%!error<ClassificationSVM: invalid Interactions parameter.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "interactions", "some")
%!error<ClassificationSVM: invalid Interactions parameter.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "interactions", -1)
%!error<ClassificationSVM: invalid Interactions parameter.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "interactions", [1 2 3 4])
%!error<ClassificationSVM: number of interaction terms requested is larger than> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "interactions", 3)
%!error<ClassificationSVM: Formula has been already defined.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "formula", "y ~ x1 + x2", "interactions", 1)
%!error<ClassificationSVM: Interactions have been already defined.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "interactions", 1, "formula", "y ~ x1 + x2")
%!error<ClassificationSVM: invalid value for Knots.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "knots", "a")
%!error<ClassificationSVM: DoF and Order have been set already.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "order", 3, "dof", 2, "knots", 5)
%!error<ClassificationSVM: invalid value for DoF.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "dof", 'a')
%!error<ClassificationSVM: Knots and Order have been set already.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "knots", 5, "order", 3, "dof", 2)
%!error<ClassificationSVM: invalid value for Order.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "order", 'a')
%!error<ClassificationSVM: DoF and Knots have been set already.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "knots", 5, "dof", 2, "order", 2)
%!error<ClassificationSVM: Tolerance must be a Positive scalar.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "tol", -1)
%!error<ClassificationSVM: ResponseName must be a char string.> ...
%! ClassificationSVM (ones(10,2), ones (10,1), "responsename", -1)


## Test input validation for predict method
%!error<ClassificationSVM.predict: too few arguments.> ...
%! predict (ClassificationSVM (ones(10,1), ones(10,1)))
%!error<ClassificationSVM.predict: Xfit is empty.> ...
%! predict (ClassificationSVM (ones(10,1), ones(10,1)), [])
%!error<ClassificationSVM.predict: Xfit must have the same number of features> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), 2)
%!error<ClassificationSVM.predict: invalid NAME in optional pairs of arguments.> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), ones (10,2), "some", "some")
%!error<ClassificationSVM.predict: includeinteractions must be a logical value.> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), ones (10,2), "includeinteractions", "some")
%!error<ClassificationSVM.predict: includeinteractions must be a logical value.> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), ones (10,2), "includeinteractions", 5)
%!error<ClassificationSVM.predict: alpha must be a scalar value between 0 and 1.> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), ones (10,2), "alpha", 5)
%!error<ClassificationSVM.predict: alpha must be a scalar value between 0 and 1.> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), ones (10,2), "alpha", -1)
%!error<ClassificationSVM.predict: alpha must be a scalar value between 0 and 1.> ...
%! predict (ClassificationSVM(ones(10,2), ones(10,1)), ones (10,2), "alpha", 'a')





