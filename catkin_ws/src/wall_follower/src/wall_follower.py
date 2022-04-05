#!/usr/bin/env python
#Ryan Reschak
#Description: Using Q-Learning, trains model to get close to a wall while
#moving along the wall

import sys
import rospy
import numpy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler

#q_table location stored
q_file = './src/wall_follower/src/q_learning2.csv'
#q_file = './q_learning.csv'

def laser_interpretor(msg):
    global state_now

    dist_angles = []
    #average of +45 and -45 degrees (the right side)
    #dist_angles.append((sum(msg.ranges[0:15])+sum(msg.ranges[345:360]))/30.0)
    dist_angles.append(min([min(msg.ranges[0:15]), min(msg.ranges[315:360])]))
    #average of +45 to +135 degrees (front side)
    #dist_angles.append(sum(msg.ranges[75:106])/30.0)
    dist_angles.append(min(msg.ranges[75:106]))
    #average of 135 to 225 degrees (the left side)
    #dist_angles.append(sum(msg.ranges[165:196])/30.0) 
    dist_angles.append(min(msg.ranges[165:226]))

    #calculate state. Wall distances Very, Medium, Far closeness
    arr_state = []
    opt_dist_wall = 0.6 #meters
    tolerance = 0.2 #meters
    
    for i in range(len(dist_angles)):
        if (i == 1):
            #Front wall
            opt_dist_wall = 1
            tolerance = 0.5
        if (i == 2):
            #Left wall
            opt_dist_wall = 0.75
            tolerance = 0.25
        
        dist = dist_angles[i]
        if (dist > opt_dist_wall-tolerance and dist < opt_dist_wall+tolerance):
            arr_state.append(1) #just right
        if(dist > opt_dist_wall): #far
            arr_state.append(2)
        else: #close
            arr_state.append(0)
    
    state_now = int(state_matrix[arr_state[0],arr_state[1],arr_state[2]])
    
    #rospy.loginfo("The current state is %d", state_now) 
    return

#Populates the state matrix
def create_state_matrix():
    state_matrix = numpy.zeros(shape=(3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                state_matrix[i,j,k] = k+j*3+i*9
    return state_matrix

def init_q_table(states, actions):
    return numpy.zeros(shape=(states, actions))

#Returns the index given the row of what action to take
def select_action(q_row, epsilon):
    n = numpy.random.random_sample()
    if (n > epsilon):
        return numpy.argmax(q_row)
    return numpy.random.randint(low=0,high=len(q_row))

#Returns reward to update Q-Table
def reward_function(state, stuck):
    state_indexes = numpy.where(state_matrix==state)
     
    if (state_indexes[0][0] == 0 or state_indexes[0][0] == 2 or state_indexes[1][0] == 0 or state_indexes[2][0] == 0):
        return -2 - stuck #Too far or too close
    elif (state_indexes[0][0] == 1):
        return 1 #Just Right amount of distance
    return -1 - stuck #exceptable

def q_value(q_table, state, action, reward, next_state, l_rate, dis_rate):
    return (1-l_rate)*q_table[state,action] + l_rate*(reward+dis_rate*max(q_table[next_state]))

def q_premade(q_table):
    #Premade Q-Table for D1
    q_table[int(state_matrix[0,0,2]), 2] = 10.0
    q_table[int(state_matrix[0,1,0]), 2] = 10.0
    q_table[int(state_matrix[2,2,2]), 0] = 10.0
    #Wall to the left
    q_table[int(state_matrix[2,1,2]), 0] = 10.0
    q_table[int(state_matrix[2,0,2]), 2] = 10.0
    #Wall to the right
    q_table[int(state_matrix[2,2,1]), 0] = 10.0
    q_table[int(state_matrix[2,2,0]), 1] = 10.0

    q_table[int(state_matrix[1,1,2]), 2] = 10.0
    q_table[int(state_matrix[1,0,2]), 2] = 10.0
    q_table[int(state_matrix[0,2,2]), 1] = 10.0
    q_table[int(state_matrix[0,2,1]), 2] = 10.0
    q_table[int(state_matrix[0,1,2]), 0] = 10.0
    q_table[int(state_matrix[0,0,2]), 2] = 10.0
    q_table[int(state_matrix[0,0,1]), 2] = 10.0
    q_table[int(state_matrix[0,0,0]), 2] = 10.0
    return q_table

def load_q_table():
    return numpy.genfromtxt(q_file, delimiter=',')

def save_q_table(q_table):
    numpy.savetxt(q_file, q_table, delimiter=',') 

def gen_pos():
    n = numpy.random.random_sample()
    if (n < 0.2):
        return [0, 0]
    elif(n < 0.4):
        return [-3,-3]
    elif(n < 0.6):
        return [3, -3]
    elif(n < 0.8):
        return [3, 3]
    else:
        return [-3, 3]

def gen_ori():
    n = numpy.random.random_sample()
    if (n < 0.25):
        return 0
    elif(n < 0.5):
        return 1.5707
    elif(n < 0.75):
        return 3.14
    else:
        return -1.5707

def test_pose(episode):
    #Will choose the location to test based on the epiosde
    n = episode % 3
    if (n == 0):
        return [[3,0],1.5707] #should go around the L
    elif (n == 2):
        return [[1,-1],0] 
    elif (n == 1):
        return [[-1,-1], 1.5707] #should go around the I
    elif (n == 3):
        return [[-2,2], 3.14]

def set_position(gazebo_set_pose_func, pos, ori):
    q = quaternion_from_euler(0, 0 , ori)
    
    state_msg = ModelState()
    state_msg.model_name = 'triton_lidar'
    state_msg.pose.position.x = pos[0] 
    state_msg.pose.position.y = pos[1]
    state_msg.pose.position.z = 0
    state_msg.pose.orientation.x = q[0]
    state_msg.pose.orientation.y = q[1]
    state_msg.pose.orientation.z = q[2]
    state_msg.pose.orientation.w = q[3]
    gazebo_set_pose_func(state_msg)

def get_position(gazebo_get_pose_func):
    state = gazebo_get_pose_func(model_name='triton_lidar')
    return numpy.array([state.pose.position.x, state.pose.position.y, state.pose.position.z]) 

def check_termination(q_table, stuck_inc, steps, max_steps):
    #Save q_table regardless
    save_q_table(q_table)
    
    #Check to see if stuck
    if (stuck_inc >= 3):
        return True    

    #Check to see if steps is greater than size
    if (steps > max_steps):
        return True
    
    return False

def main(argv):
    if (len(argv) == 0):
        print("No arguments provided")
        return
    
    global state_now
    #populates state matrix
    global state_matrix 
    state_now = 26
    state_matrix = create_state_matrix() 

    rospy.init_node("wall_follower", anonymous=True)
    
    #Service functions
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    set_bot_position = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    get_bot_position = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    pub = rospy.Publisher("triton_lidar/vel_cmd", Pose2D, queue_size=1)
    rospy.Subscriber("scan", LaserScan, laser_interpretor)
    freq = 30
    rate = rospy.Rate(freq)
    
    #Step 1: Init Q table with 27 states and 3 actions
    if (argv[0] == "learn"):
        #q_table = init_q_table(27, 3)
        #q_table = q_premade(q_table)

        q_table = load_q_table()
        rospy.loginfo("User Choose Learning") 
    else:
        q_table = load_q_table()
        rospy.loginfo("User Choose Testing") 
    
    
    #epsilon greedy values
    epsilon = 1
    eps_o = 0.9
    d = 0.985

    #q_table update values
    l_rate = 0.9
    dis_rate = 0.6
    
    #Create epsiodes and their length
    episodes = 10000
    max_steps = 3000
    while not rospy.is_shutdown():
        for episode in range(episodes):
            terminate = False
            
            #Set and get new position
            ori = None
            pos = None
            if (argv[0] == "learn"):
                #Update epsilon
                epsilon = eps_o*d**episode

                pos = gen_pos();
                ori = gen_ori();
            else:
                epsilon = 0
                max_steps = 6000
                test_obj = test_pose(episode);
                pos = test_obj[0]
                ori = test_obj[1]
            set_position(set_bot_position, pos, ori)
            prev_pos = get_position(get_bot_position)

            #Set step and stuck count
            step = 0
            stuck_inc = 0

            while not terminate:
                #Step 2: Set current state (assumes that the state_now was updated in subscriber)
                current_state = state_now
                #Step 3: Get next action
                action = select_action(q_table[current_state], 0)#, epsilon)in the works
                rospy.loginfo("The action chosen %d with state %d", action, current_state) 
                #Step 4: Execute Action based on current_state
                cmd = Pose2D() 
                const_vel = 0.3
                const_ang = 3.141519/3 #60 degrees
                #Forward, Left, Right
                if (action == 0):
                    cmd.y = const_vel
                    cmd.x = 0.0
                    cmd.theta = 0.0
                elif (action == 1): #Left
                    cmd.y = const_vel
                    cmd.x = 0.0
                    cmd.theta = -1*const_ang
                elif (action == 2): #Right
                    cmd.y = const_vel
                    cmd.x = 0.0
                    cmd.theta = const_ang
                pub.publish(cmd)

                rate.sleep()

                #Update Q table
                state_next = state_now
                reward = reward_function(state_next, stuck_inc)
                q_table[current_state, action] = q_value(q_table, current_state, action, reward, state_next, l_rate, dis_rate)

                
                #Check for Termination
                current_pos = get_position(get_bot_position)
                if (numpy.allclose(current_pos, prev_pos, atol=10e-4)):
                    stuck_inc += 1
                prev_pos = current_pos
                terminate = check_termination(q_table, stuck_inc, step, max_steps)
                step += 1
                rospy.loginfo("Step: %d. Stuck: %d", step, stuck_inc) 
        
            #Reset world
            reset_world()


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except rospy.ROSInterruptException:
        pass
