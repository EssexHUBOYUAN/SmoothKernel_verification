SmoothKernel_verification
*need 'test_file_generator.py' to generate the simulations for test

# flowwork_verification.py
veribles to be changed
head = 'C:/smoothkernel method verification/test_gensis_E/set_3/testset' # the path where the 'testset1', 'testset2', 'testset3', to be tested for the flowwork of GLF
version = '2' # version of GLF to be choosen
--------------------
the flowwork aims to get 'female pop^cor test.csv' and 'male pop^cor test.csv' which contain the TurePositive(TP), FalsePositive(FP), FalseNegtive(ND) and other metics for each population size and correlation coefficient pair.
large store size is need for this program (recommended: >40G)
# get_POPvsCORR_metrics.py
veribles to be changed
L_gensis_label=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] # the tail notation of the folder having subfolder 'set_3'
path = 'C:/smoothkernel method verification/' # the path where 'test_gensis_A' to 'test_gensis_H' folder stored
--------------------
we named the generated ingredient_used-population-counts file as test_gensis_A ... test_gensis_E.
