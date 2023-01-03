import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

import random
from sko.GA import GA


# A set of utility functions used in the features' transformers is given below

# This function outputs a tuple containing the values of a given pandas Series as a numpy array and its name as a string
def get_name(x:pd.Series):
    try:
        return x.values, x.name
    # If not a pandas Series, assume a numpy ndarray and return itself
    except AttributeError:
        print("the argument passed is not a pandas.Series")
        return x, "feat"

# This is a transformation function, applying polynomial transformation of desired degree to a pandas Series or a single-column numpy array 
# and returning a pandas DataFrame
def polynomial(x:pd.Series, degree) -> pd.DataFrame:
    # print(f"degree:  {degree}")
    values, col_name = get_name(x)
    data = values ** degree
    return pd.DataFrame(data=data, columns = [f"{col_name} - poly-{degree}"])

# This is a transformation function, applying square root transformation to a pandas Series or a single-column numpy array, while keeping the
# signs of the values, returning a pandas DataFrame
def square_root(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sign(values) * np.sqrt(np.abs(values))
    return pd.DataFrame(data=data, columns=[f"{col_name} - sqrt"])

# This function inverses the values of a pandas Series or a single-column numpy array, while accounting for zero and keeping the signs of the 
# values, retyrning a pandas DataFrame
def reciprocal(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sign(values) / (1 + np.abs(values))
    return pd.DataFrame(data=data, columns=[f"{col_name} - reciprocal"])

# This function applies Box Cox power transform to a pandas Series or a single-column numpy array, while accounting for the fact that this 
# transformation only applies for positive values, and keeping the sign. Optimal lambda value is estimated by the sklearn implementation
# of the PowerTransformer, which includes Box Cox transformation. Returns a pandas DataFrame.
def box_cox(x: pd.Series) -> pd.DataFrame:
    pt = PowerTransformer(method='box-cox', standardize=False)
    values, col_name = get_name(x)
    new_data = np.abs(values).reshape(-1, 1) + 1
    assert (new_data > 0).all()
    data = np.sign(values).reshape(-1, 1) * pt.fit_transform(new_data)
    return pd.DataFrame(data=data, columns=[f"{col_name} - box-cox"])

# This function applies Yeo-Johnson power transform to a pandas Series or a single-column numpy array. Optimal lambda value is estimated 
# by the sklearn implementation of the PowerTransformer, which includes Yeo-Johnson transformation. Returns a pandas DataFrame.
def yeo_johnson(x:pd.Series)-> pd.DataFrame:
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    values, col_name = get_name(x)
    data = pt.fit_transform(values.reshape(-1, 1))
    return pd.DataFrame(data=data, columns=[f"{col_name} - yeo-johnson"])

# This function applies quantile transform to a pandas Series or a single-column numpy array and maps it to a uniform distribution.
# Returns a pandas DataFrame.
def quantile_transformation(x:pd.Series) -> pd.DataFrame:
    qt = QuantileTransformer(random_state=0)
    values, col_name = get_name(x)
    data = qt.fit_transform(values.reshape(-1, 1))
    return pd.DataFrame(data=data, columns=[f"{col_name} - quantile"])

# This function sets the transformations that are going to be used on our features, returns a list of lambda functions.
def set_mapper(poly_degree):
    mapper = [lambda x: square_root(x),
              lambda x: reciprocal(x),
              lambda x: box_cox(x),
              lambda x: yeo_johnson(x),
                 lambda x: x]

    mapper.extend([lambda x, i=i: polynomial(x, i) for i in range(2, poly_degree + 1)])

    return mapper


# The FeatureTransformer class applies a number of functions on the features in order to improve the linear correlation between
# them and the target variables: linear relations are easily detected by machine learning models and can lead to significant
# performance boost

class FeatureTransformer:
    no_poly = 4

    # df - feature DataFrame, y - target array
    def _set_data(self, df:pd.DataFrame, y: np.array):
        self.data = df
        self.y = y

    # Initializing the genetic algorithm, variables are self-explanatory
    def __init__(self, size_pop=50, max_iter=200, prob_mut=0.001, df: pd.DataFrame=None, y:np.array=None, 
                 poly_degree:int=5, target_names:list=None):
        # Making sure either both df and y are None or both are not None
        if df is None != y is None:
            raise ValueError("Make sure that that either both data and target fields are None, or none of them")

        # Setting the data field only if the argument is explicitly passed to avoid None values
        if df:
            self.data = df
            self.y = y

         # Initialize a dictionary of the possible transformations to apply on the columns of the dataframe
        self.function_mapper = set_mapper(poly_degree)
        if target_names is None:
            self.target_names = ['y', 'target', 'dependent_variable']

        # Set the parameters for running GA
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut

        # Define a field for the latest fitted transformations
        self.fitted_x = None 
    
    # This function defines the way a chosen position in a chromosome is mutated, returns integer
    @classmethod
    def _mutation_pattern(cls, value_function, mapper):
        # If the function is among the first 4 values (non-polynomial), map it to another non-polynomial function
        num_functions = len(mapper)

        # If the function is non-polynomial, choose a new non-polynomial fucntion
        if value_function < 4:
            while True:
                new_value = random.choices(list(range(4)), k=1)[0]
                if new_value != value_function:
                    return new_value

        # If the function is polynomial, get a random value that will determine whether to increment 
        # or decrement the degree of the polynomial function
        increase_decrease_prob = random.random()
        return 4 + ((value_function + (1 if increase_decrease_prob < 0.5 else -1)) % (num_functions - 4))

    # This function randomly chooses a position inside a chromosome to mutate
    def _mutation_chromosome(self, chromosome):
        chromosome_length = len(chromosome)
        position_to_mutate = random.randint(0, chromosome_length - 1)
        chromosome[position_to_mutate] = self._mutation_pattern(chromosome[position_to_mutate], self.function_mapper)
        return chromosome

    # The actual function that will be called by the GA algorithm to perform mutation
    def _ga_mutation_function(self, algorithm):
        # According to the library's code the population is saved in a Chrom field: 2d np.array of shape (self.size_pop, self.n_dim),
        # iterate through the population
        for i in range(algorithm.size_pop):
            if np.random.rand() < algorithm.prob_mut:
                algorithm.Chrom[i] = self._mutation_chromosome(algorithm.Chrom[i])
        
        return algorithm.Chrom

    # Ensure that there is at least one target name that is not in the feature columns of the dataframe
    def _find_target_name(self):
        for name in self.target_names:
            if name not in self.data.columns:
                target = name
                return target
        # Reaching this part of the code means that all the possible names to denote the target column
        # are present in the dataframe, then raise an error
        raise ValueError("All possible target names are already in use!!!\nPlease consider adding a new target name or"
                         "\nchanging the dataframe's column names")

    def _new_features(self, chromosome: np.array, df:pd.DataFrame=None):
        if df is None:
            df = self.data

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(data=df, columns=range(df.shape[1]))
        
        # The chromosome is assumed to be a numpy array of size : number of features of the data field
        # iterate through the chromosome: each value maps to a function
        # apply this function on the corresponding column
        new_features = [self.function_mapper[int(value_function)](df[d]) for d, value_function in zip(df.columns, chromosome)]

        # Concatenate all the new features into a single dataframe
        all_data =  pd.concat(new_features, axis=1, ignore_index=False)
        
        return all_data
    
    # This function calculates the correlation between transformed features and the target
    def _get_correlation(self, chromosome) -> pd.Series:
        # Get the new features from the chromosome
        new_features = self._new_features(chromosome)
        
        # Retrieve the target name
        target_name = self._find_target_name()
        
        # Add the target variable's values as a column to the "new_features" dataframe
        new_features[target_name] = self.y.copy()
        
        # Compute the correlation matrix (linear correlation)
        linear_corr = np.abs(new_features.corr()[target_name])

        # Order the columns by their correlation to the target
        linear_corr.sort_values(ascending=False, inplace=True)
        linear_corr.drop('y', inplace=True)
        return linear_corr

    # This is the function calculating fitness value of a given chromosome by measuring correlation coefficients
    # with the target value
    def _ga_function(self, chromosome: np.array):
        linear_corr = self._get_correlation(chromosome)
        # the score is the reverse of the average score of the best "num_feats" new features
        return 1 / (linear_corr.mean())

    # This function fits a given dataset and target column into a Genetic Algorithm object from sko library
    def fit(self, df:pd.DataFrame, y: np.array):
        # If the passed object is not a dataframe, it is assumed to be a numpy array
        num_feats = df.shape[1]

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(data=df, columns=range(num_feats))
        
        # Set the data fields for later use
        self._set_data(df, y)

        # Define a function object to pass to the Genetic algorithm
        ga_function = lambda x: self._ga_function(x)

        # Define the lower and upper bounds for the chromosomes
        lower_bound = np.zeros(num_feats)
        upper_bound = np.full(shape=(num_feats, ), fill_value=len(self.function_mapper) - 1)
        # Define the precision so that values in chromosome objects are integers
        precision = np.full(shape=(num_feats, ), fill_value=1)

        # Define a GA object
        ga = GA(func=ga_function, n_dim=num_feats, size_pop=self.size_pop, max_iter=self.max_iter, prob_mut=self.prob_mut,
                lb=lower_bound, ub=upper_bound, precision=precision)

        # Register the mutation operator
        ga.register(operator_name='mutation', operator=lambda x: self._ga_mutation_function(x))

        # Run the algorithm
        best_x, best_y = ga.run()

        self.fitted_x = best_x
    
    # This function applies the tranformations onto a given DataFrame
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        if self.fitted_x is None:
            raise ValueError("The feature transformer is not fitted yet. Make sure to call FeatureTransformer.fit(X, y) beforehand")    
        
        return self._new_features(self.fitted_x, df)
    
    # This is the main function applying the GA on a set of features and a target column,
    # returning a dataset of transformed features
    def fit_transform(self, df: pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)
    
