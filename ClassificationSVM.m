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

  endproperties
