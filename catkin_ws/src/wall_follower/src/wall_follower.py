#!/usr/bin/env python
#Ryan Reschak
#Description: Using Reinforcement Learning, trains model to get close to a wall while
#moving along the wall

import rospy
import numpy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

opt_dist_wall = 2 #meters
tolerance = 0.1 #meters

def laser_interpretor(msg):
    global state_now

    dist_angles = []
    #average of +45 and -45 degrees (the right side)
    dist_angles.append((sum(msg.ranges[0:45])+sum(msg.ranges[315:360]))/90.0)
    #average of +45 to +135 degrees (front side)
    dist_angles.append(sum(msg.ranges[45:136])/90.0)
    #average of -45 to -135 degrees (the left side)
    dist_angles.append(sum(msg.ranges[225:316])/90.0)
    
    #calculate state. Wall distances Very, Medium, Far closeness
    arr_state = []
    for i in range(len(dist_angles)):
        dist = dist_angles[i]
        if (dist > opt_dist_wall-tolerance and dist < opt_dist_wall+tolerance):
            arr_state.append(1) #close
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
def reward_function(state, action):
    state_indexes = numpy.where(state_matrix==23)
    if (action == 0 and (state_indexes[1][0] == 1 or state_indexes[2][0] == 1)):
        return 10 #this means its moving forward and is next to the wall
    return -5 #Too far or Too close

def q_value(q_table, state, action, reward, next_state, l_rate, dis_rate):
    return (1-l_rate)*q_table[state,action] + l_rate*(reward+dis_rate*max(q_table[next_state]))

def main():
    global state_now
    #populates state matrix
    global state_matrix 
    state_now = 26
    state_matrix = create_state_matrix() 

    rospy.init_node("wall_follower", anonymous=True)
    #reset the gazebo world first
    #reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    #reset_world()

    pub = rospy.Publisher("triton_lidar/vel_cmd", Pose2D, queue_size=1)
    rospy.Subscriber("scan", LaserScan, laser_interpretor)
    freq = 30
    rate = rospy.Rate(freq)
    
    #Step 1: Init Q table with 27 states and 3 actions
    q_table = init_q_table(27, 3)

    #Premade Q-Table for D1
    q_table[int(state_matrix[2,2,2]), 0] = 10
    #Wall to the left
    q_table[int(state_matrix[2,1,2]), 0] = 10
    q_table[int(state_matrix[2,0,2]), 2] = 10
    #Wall to the right
    q_table[int(state_matrix[2,2,1]), 0] = 10
    q_table[int(state_matrix[2,2,0]), 1] = 10

    q_table[int(state_matrix[1,1,2]), 2] = 10
    q_table[int(state_matrix[1,0,2]), 2] = 10
    q_table[int(state_matrix[0,2,2]), 1] = 10
    q_table[int(state_matrix[0,2,1]), 2] = 10
    q_table[int(state_matrix[0,1,2]), 0] = 10
    q_table[int(state_matrix[0,0,2]), 2] = 10
    q_table[int(state_matrix[0,0,1]), 2] = 10
    q_table[int(state_matrix[0,0,0]), 2] = 10
    
    
    epsilon = 0.5 #This is the possiblity that it randomly selects an action so that it learns new things.
    l_rate = 0.2
    dis_rate = 0.4
    while not rospy.is_shutdown(): 
        #RL
        #Step 2: Set current state (assumes that the state_now was updated in subscriber)
        current_state = state_now
        #Step 3: Get next action
        action = select_action(q_table[current_state], 0)#, epsilon)in the works
        rospy.loginfo("The action chosen %d with state %d", action, current_state) 
        #Step 4: Execute Action based on current_state
        cmd = Pose2D() 
        const_vel = 0.3
        const_ang = 3.141519/3 #45 degrees
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
        #reward = reward_function(state_next, action)
        #q_table[current_state, action] = q_value(q_table, current_state, action, reward, state_next, l_rate, dis_rate)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
