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

## -*- texinfo -*-
## @deftypefn  {statistics} {@var{obj} =} fitcsvm (@var{X}, @var{Y})
## @deftypefnx {statistics} {@var{obj} =} fitcsvm (@dots{}, @var{name}, @var{value})
##
## Fit a Support Vector Machine classification model.
##
## @code{@var{obj} = fitcsvm (@var{X}, @var{Y})} returns a Support Vector Machine
## classification model, @var{obj}, with @var{X} being the predictor data,
## and @var{Y} the class labels of observations in @var{X}.
##
## @itemize
## @item
## @code{X} must be a @math{NxP} numeric matrix of input data where rows
## correspond to observations and columns correspond to features or variables.
## @var{X} will be used to train the SVM model.
## @item
## @code{Y} is @math{Nx1} matrix or cell matrix containing the class labels of
## corresponding predictor data in @var{X}. @var{Y} can contain any type of
## categorical data. @var{Y} must have same numbers of Rows as @var{X}.
## @item
## @end itemize
##
## @code{@var{obj} = fitcsvm (@dots{}, @var{name}, @var{value})} returns a
## Support Vector Machine model with additional options specified by
## @qcode{Name-Value} pair arguments listed below.
##
## @multitable @columnfractions 0.18 0.02 0.8
## @headitem @tab @var{Name} @tab @var{Value}
##
## @item @qcode{"PredictorNames"} @tab @tab A cell array of character vectors
## specifying the predictor variable names.  The variable names are assumed to
## be in the same order as they appear in the training data @var{X}.
##
## @item @qcode{"ResponseName"} @tab @tab A character vector specifying the name
## of the response variable.
##
## @item @qcode{"ClassNames"} @tab @tab A cell array of character vectors
## specifying the names of the classes in the training data @var{Y}.
##
## @item @qcode{"Cost"} @tab @tab A @math{NxR} numeric matrix containing
## misclassification cost for the corresponding instances in @var{X} where
## @math{R} is the number of unique categories in @var{Y}.  If an instance is
## correctly classified into its category the cost is calculated to be 1, If
## not then 0. cost matrix can be altered use @code{@var{obj.cost} = somecost}.
## default value @qcode{@var{cost} = ones(rows(X),numel(unique(Y)))}.
##
## @item @qcode{"Prior"} @tab @tab A numeric vector specifying the prior
## probabilities for each class.  The order of the elements in @qcode{Prior}
## corresponds to the order of the classes in @qcode{ClassNames}.
## @end multitable
##
## @seealso{ClassificationSVM}
## @end deftypefn

function obj = fitcsvm (X, Y, varargin)

  ## Check input parameters
  if (nargin < 2)
    error ("fitcsvm: too few arguments.");
  endif
  if (mod (nargin, 2) != 0)
    error ("fitcsvm: Name-Value arguments must be in pairs.");
  endif

  ## Check predictor data and labels have equal rows
  if (rows (X) != rows (Y))
    error ("fitcsvm: number of rows in X and Y must be equal.");
  endif
  ## Parse arguments to class def function
  obj = ClassificationSVM (X, Y, varargin{:});

endfunction

## Test input validation
%!error<fitcsvm: too few arguments.> fitcsvm ()
%!error<fitcsvm: too few arguments.> fitcsvm (ones (4,1))
%!error<fitcsvm: Name-Value arguments must be in pairs.>
%! fitcsvm (ones (4,2), ones (4, 1), 'Prior')
%!error<fitcsvm: number of rows in X and Y must be equal.>
%! fitcsvm (ones (4,2), ones (3, 1))
%!error<fitcsvm: number of rows in X and Y must be equal.>
%! fitcsvm (ones (4,2), ones (3, 1), 'KFold', 2)

