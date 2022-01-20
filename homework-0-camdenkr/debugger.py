# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## PROBLEM 1:

# %%
'''
Function reads dataset and returns two arrays (X and y) and turns "?" into a NaN
X = attributes for every patient
y = class for each patient
'''
def import_data(filename="./arrhythmia.data"):
    X = []
    y = []
    file = open(filename, 'r')
    lines = file.readlines()

    for line in lines:
        split_line = line.split(",")
        X.append(split_line[0:-1]) # List of lists of first 279 attributes
        y.append(split_line[-1:][0].split("\n")[0])  # Remove /n character attached to last element
  
    # Clean X and y, turn everything to floats and turn "?" into NaN
    for i, line in enumerate(X):
        for j,char in enumerate(line):
            if char == "?":
                X[i][j] = float("NaN")
            else:
                X[i][j] = float(X[i][j])

    for index, line in enumerate(y):
        y[index] = float(y[index])

    file.close()
    return X,y

# %% [markdown]
# ## PROBLEM 2
# %% [markdown]
# ### 2a)

# %%
'''
Computes median of a given list an returns it
credit: https://tinyurl.com/3x9hxxtz
'''
def compute_median(arr):
    arr.sort()
    mid = len(arr) // 2
    res = (arr[mid] + arr[~mid]) / 2
    return res

'''
Convert NaNs in X to median of feature vector (current column of X)

Method: Go through each column of X, adding each value to arr, if a NaN value is encountered at any point,
        after all data in column is added, go back to each row and replace nan with median
'''
def impute_missing(X):
    # Loop through each column
    for col in range(0,len(X[0])):
        NaN_exists = False
        arr = []
        nan_rows = []
        # Add all non-nans to array, if NaN is found, NaN_exists = True, save row it was found in nan_rows
        for row in range(0,len(X)):
            val = X[row][col]
            if(val != val):
                NaN_exists = True
                nan_rows.append(row)
            else:
                arr.append(val)
        # If there was a NaN for the feature, go back to rows and replace with median
        if (NaN_exists):
            med = compute_median(arr)
            for index in nan_rows:
                X[index][col] = med


        
    return X


        
    

# %% [markdown]
# ### 2b)
# 
# Sometimes it is better to use the median over the mean of an attribute due to **outliers** in the feature. Although the mean provides some representation of the data, it can be heavily skewed by values that are much less or greater than the rest of the values in that feature. The median is preferable to accurattely represent where the middle/center of the data lies around.
# %% [markdown]
# ### 2c)

# %%
'''
Takes in X and y and removes all rows from X that contain a NaN value, removing the
corresponding classes as well
'''
def discard_missing(X,y):
    i = 0
    while (i<len(X)):
        for j, el in enumerate(X[i]):
            # If element is a Nan, remove row completely, stay on same row index
            if (el != el):
                X.pop(i)
                y.pop(i)
                i -= 1
                break;
        i += 1
    return X,y
    
    

# %% [markdown]
# ## Problem 3
# %% [markdown]
# ### 3a)

# %%

import numpy as np
'''
Shuffles rows of data X and corresponding classes of y

Method: Create numpy arrays of X, y, generate random permutation of each,
        index X,y with same permutation and convert them back to list types        
'''
def shuffle_data(X,y):
    numpy_X = np.array(X)
    numpy_Y = np.array(y)
    p = np.random.permutation(len(numpy_X))
    X = numpy_X[p].tolist()
    y = numpy_Y[p].tolist()
    return X,y

# %% [markdown]
# ### 3b)

# %%
# Returns the standard deviation of a list
def std_form(arr):
    std = 0
    N = len(arr)
    mean = 0
    for i in range(0,N):
        mean += arr[i]
    mean /= N
    for i in range(0,N):
        std += (arr[i] - mean) ** 2
    std /= (N-1)
    std = std**.5
    return std

'''
Takes in matrix X, and computes the std deviation for each column/feature

std = list of std deviations for each feature
'''
def compute_std(X):
    std = []
    for col in range(0,len(X[0])):
        print(X)
        feature = [i[col] for i in X]
        std.append(std_form(feature))

    return std

# %%

# %% [markdown]
# ### 3c)

# %%
'''
Takes in matrix X, and computes the mean for each column/feature

mean = list of averages (avg) for each feature
'''
def compute_mean(X):
    mean = []
    for col in range(0,len(X[0])):
        feature = [i[col] for i in X]
        avg = 0
        for el in feature:
            avg += el
        avg /= len(feature)
        mean.append(avg)

    return mean

'''
Removes all rows of data that have an attribute that is greater 2*σ
'''
def remove_outlier(X,y):
    # Get list of standard deviations for each feature
    std = compute_std(X)
    mean = compute_mean(X)
    i = 0
    # While the current row index is less than the total number of rows left,
    # continue iterating through the columns of current row checking attributes
    while (i<len(X)):
        for j in range(0,len(X[i])):
            el = X[i][j]
            # Remove row if attribute more than 2 standard deviations from mean
            if (el > mean[j]+(2 * std[j])):
                X.pop(i)
                y.pop(i)
                i -= 1
                break;
        i += 1
    return X,y

# %% [markdown]
# ### 3d)
# 
# Time complexity: O(n*m) where n is the size of X and m is the size of each row of X. compute_std(), compute_mean(), and the rest of standardize_data() all iterate over each element in each row of data once.
# 
# Space Complexity: O(n+m) where n is the size of X and m is the size of each row of X. To compute the mean and standard deviation, we store arrays of each column (size n) and we store the final arrays of stdevs and means of each column which is size of the number of columns (size m). If we look at only the function itself after getting the std deviation and mean lists, then it would be O(m) only.

# %%
'''
Standardizes data in X and returns it back.
'''
def standardize_data(X):
    std = compute_std(X)
    mean = compute_mean(X)
    N = len(X[0]) 
    for i in range(0,len(X)):
        for j in range(0,N):
            # Standardize where stdev is 0 by replacing value with 0
            if (std[j]==0):
                X[i][j] = 0
            else:
                X[i][j] -= mean[j]
                X[i][j] /= std[j]
    return X

# %% [markdown]
# Main function for problems 1-2:

# %%
X,y = import_data("/Users/chrisjr38/Desktop/cs506_hmwk/homework-0-camdenkr/arrhythmia.data")
# X = impute_missing(X)
X,y = discard_missing(X,y)
# # X,y = shuffle_data(X,y)
# std = compute_std(X)
X, y = remove_outlier(X,y)
# X = standardize_data(X)

# %% [markdown]
# ## Problem 4
# %% [markdown]
# 

# %%
'''
Takes in titanic dataset from "./train.csv" and extracts and cleans data into matrix X and class y
'''

def import_titanic_dataset(filename="./train.csv"):
    X = []
    y = []
    f = open(filename, 'r')
    lines = f.readlines()
    # Column j=1 is the class so don't include it in X
    # Column j=3 is a comma separated name, so include both parts of split back together
    for line in lines[1:]:
        split_line = line.split(',')
        parsed_line = []
        # Append columns 0, 2
        parsed_line.append(split_line[0])
        parsed_line.append(split_line[2])
        # Recombine split name and append
        name = split_line[3] + ", " + split_line[4]
        parsed_line.append(name)
        # Append the rest of the columns
        i = 5
        while(i < len(split_line)):
            # Convert sex (i=5) into 0 if female, 1 if male
            if (i==5):
                if (split_line[i] == ""):
                    sex = float("NaN")
                else:
                    sex = 0 if (split_line[i] == "female") else 1
                parsed_line.append(sex)
                i+=1
                continue;
            # Convert embarked (j = last column) such that C=0,Q=1, S=2 -- NaN if blank
            if (i==len(split_line)-1):
                embarked = split_line[i].split("\n")[0]  # Remove /n character attached to last element
                if (embarked == "C"):
                    embarked = 0
                elif (embarked == "Q"):
                    embarked = 1
                elif (embarked == "S"):
                    embarked = 2
                else:
                    embarked = float("NaN")
                parsed_line.append(embarked)
                i+=1
                continue;
            parsed_line.append(split_line[i])
            i += 1
        X.append(parsed_line)
        
    # Take survived column of each line and add it to y (converted to float)
    for line in lines[1:]:
        y.append(float(line.split(",")[1]))

    f.close()
    return X,y

# %% [markdown]
# main function for problem 4:

# %%
# X,y = import_titanic_dataset("./train.csv")

# %% [markdown]
# ## Problem 5
# %% [markdown]
# ### 5a)

# %%
import numpy as np
'''
Takes in data X, labels y, and decimal t_f which indicates
how much of the dataset should be randomly split into test set,
with the remaining going to a training set

Method: shuffles X and y randomly, then simply takes splits the first t_f into test and other t_f into train

'''
def train_test_split(X, y, t_f):
    X_train, y_train, X_test, y_test = [], [], [], []
    # Shuffle the data
    X, y = shuffle_data(X,y)
    assert(len(X) == len(y))
    # Get index where to split the data based on percentage of total number of valuess
    part_index = int(len(X)*t_f)
    # Partition data around part_index
    X_test = X[0:part_index]
    y_test = y[0:part_index]
    X_train = X[part_index:]
    y_train = y[part_index:]
    return X_train, y_train, X_test, y_test

# %% [markdown]
# ### 5b)

# %%
'''
Splits the dataset into 3 sections, test, training, and cross validation sets and returns them
'''
def train_test_CV_split(X, y, t_f, cv_f):
    X_train, y_train, X_test, y_test, X_cv, y_cv = [], [], [], [], [], []
    # Shuffle the data
    X, y = shuffle_data(X,y)
    assert(len(X) == len(y))
    
    # Assign index based on first split around test set
    part_index_t_f = int(len(X)*t_f)
    # Assign second partition index from the previous index
    part_index_cv_f = int(len(X)*cv_f)+part_index_t_f
    '''Partition data from beginning to part_index_t_f to go to testing set
    then put from part_index_t_f to part_index_cv_f to go to cross validation set
    then put the rest into the training set''' 
    X_test = X[0:part_index_t_f]
    y_test = y[0:part_index_t_f]
    X_cv = X[part_index_t_f:part_index_cv_f]
    y_cv = y[part_index_t_f:part_index_cv_f]
    X_train = X[part_index_cv_f:]
    y_train = y[part_index_cv_f:]
    return X_train, y_train, X_test, y_test, X_cv, y_cv

# %% [markdown]
# main function for problem 5 (keep X and y assignments commented to use previous X and y matrices)

# %%
# X = [[1,2],[3,4],[5,6],[7,8],[9,10]]
# y = [1,2,3,4,5]
# t_f = 2/10
# X_train, y_train, X_test, y_test = train_test_split(X,y,t_f)
# cv_f = 4/10
# X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_CV_split(X,y,t_f,cv_f)


