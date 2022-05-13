#!/usr/bin/env python3
#Ryan Reschak
#Description: Using Depp Q-Learning, trains model to get close to a wall while
#moving along the wall

import sys
import rospy
import numpy as np
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
#from tf.transformations import quaternion_from_euler
from NN import DQL

#q_table location stored
#q_file = './deep_q_learning'
#This is the best Q table file
q_file = './q_testing.weight.ww'
reward_file = './reward_q.csv'

def laser_interpretor(msg):
    global state_now
    global lidar_data
    
    #averages lidar data every 5 degrees
    lidar_data = np.average(np.array(msg.ranges[0:360]).reshape(-1, 15), axis=1).reshape(1, int(360/15))

    dist_angles = []
    #average of +45 and -45 degrees (the right side)
    #dist_angles.append((sum(msg.ranges[0:15])+sum(msg.ranges[345:360]))/30.0)
    dist_angles.append(min([min(msg.ranges[0:15]), min(msg.ranges[345:360])]))
    #average of +45 to +135 degrees (front side)
    #dist_angles.append(sum(msg.ranges[75:106])/30.0)
    dist_angles.append(min(msg.ranges[70:111]))
    #average of 135 to 225 degrees (the left side)
    #dist_angles.append(sum(msg.ranges[165:196])/30.0) 
    dist_angles.append(min(msg.ranges[165:196]))

    #calculate state. Wall distances Very, Medium, Far closeness
    arr_state = []
    opt_dist_wall = 0.6 #meters
    tolerance = 0.2 #meters
    
    for i in range(len(dist_angles)):
        if (i == 1):
            #Front wall
            #opt_dist_wall = 1
            #tolerance = 0.5
            opt_dist_wall = 0.75
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
    #Designed for first filled with initial state
    #Once filled will only add 3 at time
    '''for i in range(3):
        lidar_data.pop(0)
    lidar_data.append(dist_angles[0])
    lidar_data.append(dist_angles[1])
    lidar_data.append(dist_angles[2])'''
    
    state_now = int(state_matrix[arr_state[0],arr_state[1],arr_state[2]])
    '''for i in range(3):
        lidar_data.pop(0)
    lidar_data.append(arr_state[0])
    lidar_data.append(arr_state[1])
    lidar_data.append(arr_state[2])'''

    #rospy.loginfo("The current state is %d", state_now) 
    return

#Populates the state matrix
def create_state_matrix():
    state_matrix = np.zeros(shape=(3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                state_matrix[i,j,k] = k+j*3+i*9
    return state_matrix

#Returns reward to update Q-Table
def reward_function(state, stuck, action):
    state_indexes = np.where(state_matrix==state)
     
    if (state_indexes[0][0] == 0 or state_indexes[0][0] == 2 or state_indexes[1][0] == 0 or state_indexes[2][0] == 0):
        return -10 #Too far or too close
    elif (state_indexes[0][0] == 1):
        if (action == 0):
            return 10
        return 0 #Just Right amount of distance
    return -2 #exceptable

def save_q_table(model):
    return model.save(q_file)

def load_q_table(model):
     model.load(q_file)

def gen_pos():
    n = np.random.random_sample()
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
    n = np.random.random_sample()
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
    #q = quaternion_from_euler(0, 0 , ori)
    
    state_msg = ModelState()
    state_msg.model_name = 'triton_lidar'
    state_msg.pose.position.x = pos[0] 
    state_msg.pose.position.y = pos[1]
    state_msg.pose.position.z = 0
    state_msg.pose.orientation.x = 0#q[0]
    state_msg.pose.orientation.y = 0#q[1]
    state_msg.pose.orientation.z = 0#q[2]
    state_msg.pose.orientation.w = 0#q[3]
    gazebo_set_pose_func(state_msg)


def get_position(gazebo_get_pose_func):
    state = gazebo_get_pose_func(model_name='triton_lidar')
    return np.array([state.pose.position.x, state.pose.position.y, state.pose.position.z]) 

def check_termination(stuck_inc, steps, max_steps):
    
    #Check to see if stuck
    if (stuck_inc >= 3):
        return True    

    #Check to see if steps is greater than size
    if (steps > max_steps):
        return True
    
    return False

def init_experiment_table(num_episodes):
    return np.zeros(shape=(num_episodes,2))

def save_experiment_results(table):
    np.savetxt(reward_file, table, delimiter=',')

def main(argv):
    if (len(argv) == 0):
        print("No arguments provided")
        return
    
    global state_now
    global lidar_data
    state_now = 26
    lidar_data = []
    #for i in range(3):
     #   lidar_data.append(2)

    #populates state matrix
    global state_matrix 
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
    
    #Create epsiodes and their length
    episodes = 10000
    max_steps = 3000
    
    exp_table = init_experiment_table(episodes)

    #Step 1: Init Q table with 27 states and 3 actions
    model = None
    if (argv[0] == "learn"):
        model = DQL(int(360/15), 3, epsilon=1, decay=0.925)
        rospy.loginfo("User Choose Learning") 
    else:
        model = DQL(3, 3, epsilon=0)
        load_q_table(model)
        
        rospy.loginfo("User Choose Testing") 
    
    #Used in replay_memory
    batch_size = 300

    while not rospy.is_shutdown():
        for episode in range(episodes):
            terminate = False
            
            #Set and get new position
            ori = None
            pos = None
            if (argv[0] == "learn"):
                #Update epsilon
                pos = gen_pos();
                ori = gen_ori();
            else:
                test_obj = test_pose(episode);
                pos = test_obj[0]
                ori = test_obj[1]
            set_position(set_bot_position, pos, ori)
            prev_pos = get_position(get_bot_position)

            #Set step and stuck count
            step = 0
            stuck_inc = 0 #incrementer
            stuck_change = False #change stuck
            
            while not terminate:

                #Step 2: Set current state (assumes that the state_now was updated in subscriber)
                current_state = lidar_data#np.array(lidar_data)[:3].reshape(1,3) #This is used for model 
                #print(current_state)
                #Step 3: Get next action
                action = model.action_choice(current_state)
                rospy.loginfo("The action chosen %d with state %d", action, state_now) 
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
                
                if ( argv[0] == "learn" ):
                    state_next = lidar_data#np.array(lidar_data)[:3].reshape(1,3)
                    #Uses state_now globabl variable because its discretized and easier to interpret for appropriate reward
                    reward = reward_function(state_now, stuck_inc, action)

                    model.update_replay_memory(current_state, action, reward, state_next)
                 

                #Check for Termination
                current_pos = get_position(get_bot_position)
                if (np.allclose(current_pos, prev_pos, atol=10e-4)):
                    stuck_inc += 1
                    stuck_change = True
                prev_pos = current_pos
                terminate = check_termination(stuck_inc, step, max_steps)
                step += 1
                rospy.loginfo("Step: %d. Stuck: %d", step, stuck_inc)
                #Update Experiment table
                exp_table[episode,0] += reward
                exp_table[episode,1] = stuck_inc

            if len(model.replay_memory) > batch_size*3 and argv[0] == "learn":
                print("Replaying")
                model.replay(batch_size, episode)
                if episode % 50 == 0:
                    save_q_table(model)
    
            #Save Experiment info
            save_experiment_results(exp_table)        
            #Reset world
            reset_world()


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except rospy.ROSInterruptException:
        pass
