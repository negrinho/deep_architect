### ${MARKDOWN} 

# Search Space Tutorial 
# **********************
#
#  
# What is search space 
# ====================
# 
# 
# Example: Simple search space 
# =============================
# 1. Basline MLP architecture 
# 2. Visual of a simple MLP search space 
# 3. DeepArchitect code representing the search space 
# 
# DeepArchitect Search Space Basics  
# =================================
# We will explain the building blocks of search space, in the context of the 
# MLP search space above 
# 1. Hyperparameter
# 2. Modules
# 3. Compile function 
# 4. Forward function 
# 5. Dictionary of hyperparameter
# 6. Single-input, single-ouput
# 7. Pre-defined modules: 
#   - sequential
#   - optional
#   - permutation 
# 
# Designing Your Own Search Space 
# =======================
# 1. Write your own modules: CNN 
# 2. Use a pre-written modules in general search space 
# 3. Construct a CNN search space
# 4. Advanced Search Space: Unet and SubstitutionModule 
# 
# More Examples
# =============
# 1. NAS
# 2. Unet
# 3. CNN
# 4. RNN
# 5. CNN-RNN hybrid 


### ${CODE} 
def dnn(): 
    return mo.siso_sequential([
        affine(D([num_classes])), 
        nonlinearity(D(['relu', 'tanh']))
    ])
