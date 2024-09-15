###################################################
##             ASU CSE 571 ONLINE                ##
##        Unit 3 Reasoning under Uncertainty     ##
##             Project Submission File           ##
##                    agent.py                   ##
###################################################

###################################################
##                !!!IMPORTANT!!!                ##
##        This file will be auto-graded          ##
##    Do NOT change this file other than at the  ##
##       Designated places with your code        ##
##                                               ##
##  READ the instructions provided in the code   ##
###################################################

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

def buildDBN():

    # Construct a DBN object
    dbn = DBN()
    
    #!!!!!!!!!!!!!!!  VERY IMPORTANT  !!!!!!!!!!!!!!!
    # MAKE SURE to NAME THE RANDOM VARIABLE "LOCATION AS "L" AND "SENSOR" OBSERVATION AS "O"
    ########-----YOUR CODE STARTS HERE-----########



    
    ########-----YOUR CODE ENDS HERE-----########
    
    # Do NOT forget to initialize before doing any inference! Otherwise, errors will be introduced.
    dbn.initialize_initial_state()
 
    # Create an inference object
    dbn_inf = DBNInference(dbn)

    ########-----YOUR MAY TEST YOUR CODE BELOW -----########
    ########-----ADDITIONAL CODE STARTS HERE-----########




    ########-----YOUR CODE ENDS HERE-----########
    
    return dbn_inf
