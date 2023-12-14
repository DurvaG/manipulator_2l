import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from collision_check import line_circle_intersection

# # Global Variables
start_point = np.array([1, 1.7])
goal = np.array([0.1, -0.15])
obstacle = np.array([0.6, 1])

# Potential parameters
obstacle_potential = 0.1
goal_potential = 10

# Gradient Descent Parameters
learning_rate = 5E-4

# Link Parameters
l1 = 1.0
l2 = 1.0
DT = 0.1

horizon_length = 10
angular_rate = 0.2
number_of_states = 4
number_of_control_inputs = 2
R = np.diag([0.5, 0.5])  # input cost matrix
Q = np.diag([100.0, 100.0, 0.0, 0.0])  # state cost matrix#MPC helper
windowSize = 5


#Plotter setup  
plt.close('all')
#fig, ax = plt.subplots()

plt.ion()  
fig, ax = plt.subplots()
plt.axis('equal')
plt.axis([-1.0*windowSize, windowSize, -1.0*windowSize, windowSize])
fig_workspace = plt.Circle((0, 0), 2.0, color='blue', alpha = 0.1)
ax.add_patch(fig_workspace)

#Plot goal
fig_goal_handle, = plt.plot(0, 0, 'go', ms = 10.0)

#Plot obs
fig_obs_handle, = plt.plot(0, 0, 'ko', ms = 10.0)

fig_traj_handle, = plt.plot(0, 0, 'b', ms = 2)


#Plot robot as such
# plt.plot(0, 0, 'ko', ms = 4.0)
fig_cur_robot_dot= []
temp, = plt.plot(0, 0, 'ro', ms = 4.0)
fig_cur_robot_dot.append(temp)
temp, = plt.plot(0, 0, 'ro', ms = 4.0)
fig_cur_robot_dot.append(temp)
fig_cur_robot_line = []
temp, = plt.plot([0, 0], [0, 0], '-r', alpha = 1.0)
fig_cur_robot_line.append(temp)
temp, = plt.plot([0, 0], [0, 0], '-r', alpha = 1.0)
fig_cur_robot_line.append(temp)

def plot_obs_and_goal(goal, obs, trajectory):
    #Plot goal
    fig_goal_handle.set_xdata(goal[0])
    fig_goal_handle.set_ydata(goal[1])

    #Plot obstacle
    fig_obs_handle.set_xdata(obs[0])
    fig_obs_handle.set_ydata(obs[1])

    fig_traj_handle.set_xdata(trajectory[:,0])
    fig_traj_handle.set_ydata(trajectory[:,1])
    
    return

def plot_robo(thetas):
    # fig = plt.figure()
    plt.plot(0, 0, 'ko', ms = 4.0)
    #Plot robot as such
    fig_cur_robot_dot[0].set_xdata(l1*math.cos(thetas[0]))
    fig_cur_robot_dot[0].set_ydata(l1*math.sin(thetas[0]))
    fig_cur_robot_dot[1].set_xdata(l1*math.cos(thetas[0]) + l2*math.cos(thetas[0]+thetas[1]))
    fig_cur_robot_dot[1].set_ydata(l1*math.sin(thetas[0]) + l2*math.sin(thetas[0]+thetas[1]))
    
    fig_cur_robot_line[0].set_xdata([0.0, l1*math.cos(thetas[0])])
    fig_cur_robot_line[0].set_ydata([0.0, l1*math.sin(thetas[0])])
    fig_cur_robot_line[1].set_xdata([l1*math.cos(thetas[0]), l1*math.cos(thetas[0]) + l2*math.cos(thetas[0]+thetas[1])])
    fig_cur_robot_line[1].set_ydata([l1*math.sin(thetas[0]), l1*math.sin(thetas[0]) + l2*math.sin(thetas[0]+thetas[1])])
    fig.canvas.draw()
    fig.canvas.flush_events()
    ax.set_title('Setup')

    plt.show()
    return



def inv_kin(goal):
    xd = goal[0]
    yd = goal[1]

    #solve
    c_theta2 = (((xd**2) + (yd**2)  - (l1**2) - (l2**2)) / (2*l1*l2))
    s_theta2 = math.sqrt(1.0 - (c_theta2**2))
    theta2 = math.atan2(s_theta2,c_theta2)
    M = l1 + l2*math.cos(theta2)
    N = l2*math.sin(theta2)
    gamma = math.atan2(N,M)
    theta1 = math.atan2(yd,xd) - gamma
    return [theta1, theta2]


# Define the potential function
def calculate_potential(X, Y):    
    return ((1/((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2)) * obstacle_potential + ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2) * goal_potential)

# Defining the function to calculate the gradient
def gradient_f(X, Y):
    try:
        Fxg = -goal_potential*(X - goal[0])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5)

        Fyg = -goal_potential*(Y - goal[1])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5)
        Fxo = obstacle_potential*(X - obstacle[0])/((((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
        Fyo = obstacle_potential*(Y - obstacle[1])/((((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
    except RuntimeWarning:
        Fxg = 0
        Fyg = 0
    Fx = Fxg + Fxo
    Fy = Fyg + Fyo

    return np.array([Fx, Fy])

# Gradient descent algorithm
def gradient_descent(gradient, start, learning_rate):
    trajectory = [start]
    current = start
    while(math.dist(current, goal) > 0.01):
        grad = gradient(*current)
        current = current + learning_rate * grad
        trajectory.append(current)
        if math.dist(current, goal) < 0.01:
            print("Path Planning Success")
            break            
    
    return np.array(trajectory)

# Plot the potential field on a 3D subplot
def plot_potential(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis (Constant)')
    ax.set_title('3D Potential Field')
    plt.show()
    return


def get_nparray_from_matrix(x):
    return np.array(x).flatten()

#The motion model for MPC
def get_linear_model_matrix(theta_bar, U_bar):

    theta_1 = theta_bar[0]
    theta_2 = theta_bar[1]
    theta_1_dot = U_bar[0]
    theta_2_dot = U_bar[1]    

    A = np.zeros((number_of_states, number_of_states))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0    
    A[0, 2] = -1.0*(((l1*math.cos(theta_1) + l2*math.cos(theta_1+theta_2))*theta_1_dot) + (l2*math.cos(theta_1+theta_2)*theta_2_dot))*DT
    A[0, 3] = -1.0*(((l2*math.cos(theta_1+theta_2))*theta_1_dot) + (l2*math.cos(theta_1+theta_2)*theta_2_dot))*DT
    A[1, 2] = -1.0*(((l1*math.sin(theta_1) + l2*math.sin(theta_1+theta_2))*theta_1_dot) + (l2*math.sin(theta_1+theta_2)*theta_2_dot))*DT
    A[1, 3] = -1.0*(((l2*math.sin(theta_1+theta_2))*theta_1_dot) + (l2*math.sin(theta_1+theta_2)*theta_2_dot))*DT    

    B = np.zeros((number_of_states, number_of_control_inputs))
    B[0, 0] = -1.0*(l1*math.sin(theta_1) + l2*math.sin(theta_1+theta_2))*DT
    B[0, 1] = -1.0*(l2*math.sin(theta_1+theta_2))*DT
    B[1, 0] = (l1*math.cos(theta_1) + l2*math.cos(theta_1+theta_2))*DT
    B[1, 1] = (l2*math.cos(theta_1+theta_2))*DT
    B[2, 0] = DT
    B[3, 1] = DT    

    C = np.zeros(number_of_states)
    C[0] = (((l1*math.cos(theta_1) + l2*math.cos(theta_1+theta_2))*theta_1*theta_1_dot) + (l2*math.cos(theta_1+theta_2))*((theta_2_dot*(theta_1 - theta_2)) - (theta_1_dot*theta_2)))*DT
    C[1] = (((l1*math.sin(theta_1) + l2*math.sin(theta_1+theta_2))*theta_1*theta_1_dot) + (l2*math.sin(theta_1+theta_2))*((theta_2_dot*(theta_1 - theta_2)) - (theta_1_dot*theta_2)))*DT
    
    return A, B, C


#The MPC implimentation
def mpc(X_ref, X_bar, U_bar):    
    
    #Create the optimsation variable x and u
    #Argument is the shape of the vector
    x = cvxpy.Variable((number_of_states, horizon_length + 1))
    u = cvxpy.Variable((number_of_control_inputs , horizon_length))    
    
    #set up costs
    cumulative_cost  = 0.0
    for t in range (horizon_length):        
        #Add up control cost
        cumulative_cost += cvxpy.quad_form(u[:, t], R)    
        
        #Add up state cost for updates cycles
        if t != 0:
            cumulative_cost += cvxpy.quad_form(X_ref[:, t] - x[:, t], Q)    
            
    #Add up state cost for last update cycle
    cumulative_cost += cvxpy.quad_form(X_ref[:, horizon_length] - x[:, horizon_length], Q)    
    
    #set up constraints
    constraint_vector = []    
    
    for t in range(horizon_length):        
        
        #Get updated model matrices
        A, B, C = get_linear_model_matrix([X_bar[2],X_bar[3]], U_bar)        
        
        #Add state evolution constraint
        constraint_vector += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]        
        
        #Add control constraint
        constraint_vector += [cvxpy.abs(u[:,t]) <= (angular_rate)]    

    #initial condition
    constraint_vector += [x[:, 0] == X_bar]    
    
    #Formulate problem and solve
    prob = cvxpy.Problem(cvxpy.Minimize(cumulative_cost), constraint_vector)
    prob.solve(solver=cvxpy.ECOS, verbose=False)    
    
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        OX = get_nparray_from_matrix(x.value[0, :])
        OY = get_nparray_from_matrix(x.value[1, :])
        OTheta_1 = get_nparray_from_matrix(x.value[2, :])
        OTheta_2 = get_nparray_from_matrix(x.value[3, :])
        OTheta_1_dot = get_nparray_from_matrix(u.value[0, :])
        OTheta_2_dot = get_nparray_from_matrix(u.value[1, :])    
        
    else:
        print("Error: Cannot solve mpc..")
        OX, OY, OTheta_1, OTheta_2 = None, None, None, None
        OTheta_1_dot, OTheta_2_dot = None, None    
    
    return [ OX, OY, OTheta_1, OTheta_2 ]


def main():
    # Set up the Workspace
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Calculating the potential
    Z = calculate_potential(X, Y)

    # Global Variables
    start_point = np.array([1, 1.7])
    goal = np.array([0.1, -0.15])
    obstacle = np.array([0.6, 1])
    theta_start = inv_kin(start_point)

    # Plot the contour
    plot_potential(X, Y, Z)

    # Perform gradient descent to determine the trajectory
    trajectory = gradient_descent(gradient_f, start_point, learning_rate)


    # # Plot the contour and the trajectory
    fig = plt.figure()
    cf = plt.contourf(X, Y, Z, levels=100)
    plt.plot(trajectory[:,0], trajectory[:,1], 'ro-', ms=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent on Contour Plot')
    fig.colorbar(cf)
    plt.show()

    

    plot_obs_and_goal(goal, obstacle, trajectory)
    plot_robo([theta_start[0], theta_start[1]])


    j = 0
    while(math.dist(start_point, goal) > 0.05):
        #latch down initial pose
        X_bar = np.zeros(number_of_states)
        X_bar[0] = start_point[0]
        X_bar[1] = start_point[1]
        X_bar[2] = theta_start[0]
        X_bar[3] = theta_start[1]

        #latch down initial controls
        U_bar = np.zeros(number_of_control_inputs)    
        
    

        X_ref = np.zeros((number_of_states, horizon_length + 1))
        for i in range(horizon_length + 1):
            X_ref[0,i] = trajectory[i+j,0]
            X_ref[1,i] = trajectory[i+j,1]
 
        # Collision check of the link
        if not line_circle_intersection(l1*math.cos(theta_start[0]), l1*math.sin(theta_start[0]), 
                                    l1*math.cos(theta_start[0]) + l2*math.cos(theta_start[0]+theta_start[1]),
                                    l1*math.sin(theta_start[0]) + l2*math.sin(theta_start[0]+theta_start[1]), obstacle[0], obstacle[1],
                                    r=0.2):
            #Run mpc control
            print("Link 2 does not intersect with the circle")
            OX, OY, OTheta_1, OTheta_2 = mpc(X_ref, X_bar, U_bar)    
            print (OX[1])
            print (OY[1])    
            plot_robo([OTheta_1[1], OTheta_2[1]]) 


            time.sleep(0.1)

            # Updates
            start_point = [OX[1], OY[1]]
            theta_start = [OTheta_1[1], OTheta_2[1]]
            j=j+1
        else:
            input("Link 2 intersects. Press any key to terminate")
            exit()    
    return


    
#run
if __name__ == '__main__':    
    main()
    print ('Program ended')