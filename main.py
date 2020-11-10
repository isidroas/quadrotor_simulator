#!/bin/env python3
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = True
import time
from pdb import set_trace
import os
import datetime


# Parameters
DATA_L = 1000
MASS = 1  # Kg
G_CONSTANT = 9.8  # m/s^2
INERTIA = MASS * 0.45 ** 2 / 12  # Kg.m^2
DT = 0.01  # s # Reducir el paso mejora la precisión de la predicción, aunque haya ruido

# ACCEL_NOISE = 0.35 # m/s^2
ACCEL_NOISE = 1  # m/s^2
# GYRO_NOISE = 0.015 # rad/s
GYRO_NOISE = 0.03  # rad/s
GPS_NOISE= 0.7 # m
#GPS_NOISE = 0.0  # m
#GPS_DELAY = 0.2  # s
GPS_DELAY = 0  # s
VISION_NOISE = 0.05  # m

# Plot flags
DRAW_ESTIMATED = True
SHOW_ANIMATED = False
SHOW_PLOTS = False
SHOW_OUTPUT_FILT = True
IMAGE_FOLDER = "images/"
IMAGE_EXTENSION = "png"

# Fusion flags
FUSE_GPS = True
HANDLE_DELAYS = False
TAU_P = 1 
TAU_V = 1 
TAU_theta = 1 


# Control se realiza sobre los estados reales para acotar más el efecto del estimador
def control_actuators(
    theta: float, thetad: float, theta_ref: float, yd_e: float
) -> [float, float]:
    # Control gains
    K_height = 2
    K_tilt = 0.2
    Kd_tilt = 0.1
    thrust = MASS * G_CONSTANT / np.cos(theta) + yd_e * K_height
    torque = (theta_ref - theta) * K_tilt - thetad * Kd_tilt
    return thrust, torque



# Jacobianos de los modelos de observación
H_vision = np.zeros((2, 5))
H_vision[0, 0] = 1
H_vision[1, 1] = 1
R_vision = np.diag([VISION_NOISE ** 2, VISION_NOISE ** 2])

H_gps = np.zeros((2, 5))
H_gps[0, 0] = 1
H_gps[1, 1] = 1
R_gps = np.diag([GPS_NOISE ** 2, GPS_NOISE ** 2])

def output_filter( 
    p_prev, v_prev, theta_prev, accel, gyro
) -> list:

    if np.isnan(p_prev[0]):
        # we need to initialize
        p_prev = np.array([0,0])
        v_prev = np.array([0,0])
        theta_prev = 0

    theta_pred = theta_prev + DT * gyro
    #c = np.cos(theta_pred)
    #s = np.sin(theta_pred)
    # TODO: utilizar aquí el predicho ahora o el estimado anterior?
    c = np.cos(theta_prev)
    s = np.sin(theta_prev)
    rot_mat = np.array([[c, -s], [s, c]])
    v_pred = v_prev + DT * rot_mat @ accel
    p_pred = p_prev + DT * v_prev
    return p_pred, v_pred, theta_pred
    

def ekf_estimator(
    p_prev, v_prev, theta_prev, cov_mat_prev, accel, gyro, gps=None
) -> list:

    if gyro is None:
        # we can't predict states without IMU measurements
        return [None]*4

    if np.isnan(p_prev[0]):
        # we need to initialize
        p_prev = np.array([0,0])
        v_prev = np.array([0,0])
        theta_prev = 0
        cov_mat_prev = np.zeros([5,5])# TODO: esto está bien?
        
    # Predicción de los estados
    theta_pred = theta_prev + DT * gyro
    #c = np.cos(theta_pred)
    #s = np.sin(theta_pred)
    # TODO: utilizar aquí el predicho ahora o el estimado anterior?
    c = np.cos(theta_prev)
    s = np.sin(theta_prev)
    rot_mat = np.array([[c, -s], [s, c]])
    v_pred = v_prev + DT * rot_mat @ accel
    p_pred = p_prev + DT * v_prev

    # Predicción de la matriz de covarianzas
    x_pred = np.array([p_pred[0], p_pred[1], v_pred[0], v_pred[1], theta_pred])
    F = np.array(
        [
            [1, 0, DT, 0, 0],
            [0, 1, 0, DT, 0],
            [
                0,
                0,
                1,
                0,
                DT * (-accel[0] * np.sin(theta_prev) + accel[1] * np.cos(theta_prev)),
            ],
            [
                0,
                0,
                0,
                1,
                DT * (-accel[0] * np.cos(theta_prev) - accel[1] * np.sin(theta_prev)),
            ],
            [0, 0, 0, 0, 1],
        ]
    )
    G = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],  # que pasaría si desarrollo v(a) aquí?
            [DT * c, DT * s, 0],
            [-DT * s, DT * c, 0],
            [0, 0, DT],
        ]
    )
    Q = (
        G
        @ np.diag([ACCEL_NOISE ** 2, ACCEL_NOISE ** 2, GYRO_NOISE ** 2])
        @ np.transpose(G)
    )
    cov_mat_est = F @ cov_mat_prev @ np.transpose(F) + Q

    x_est = x_pred
    p_est = x_est[0:2]  # Remind slices x:y doesn't include y
    v_est = x_est[2:4]
    theta_est = x_est[4]
    cov_mat = cov_mat_est

    ### Update

    ## gps
    if FUSE_GPS and not np.isnan(gps).any():
        innov = gps - p_est
        S_gps = H_gps @ cov_mat @ np.transpose(H_gps) + R_gps
        K_f = cov_mat @ np.transpose(H_gps) @ np.linalg.inv(S_gps)
        x_est = x_est + K_f @ innov
        p_est = x_est[0:2]  # Remind slices x:y doesn't include y
        v_est = x_est[2:4]
        theta_est = x_est[4]
        cov_mat = cov_mat - K_f @ H_gps @ cov_mat

    return p_est, v_est, theta_est, cov_mat


def draw_animation(x, y, theta):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], "r")
    ln2, = plt.plot([], [], "b")

    def init():
        margin = 2
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_ylim(min(y) - margin, max(y) + margin)
        ax.set_aspect("equal")
        return (ln,)

    def update(frame):
        xdata.append(x[frame])
        ydata.append(y[frame])
        ln.set_data(xdata, ydata)
        c = np.cos(theta[frame])
        s = np.sin(theta[frame])
        rot_mat = np.array([[c, -s], [s, c]])
        p1 = [-0.5, 0]
        p2 = [0.5, 0]
        p1_rot = rot_mat @ p1
        p2_rot = rot_mat @ p2
        ln2.set_data(
            [p1_rot[0], p2_rot[0]] + x[frame], [p1_rot[1], p2_rot[1]] + y[frame]
        )
        return ln, ln2

    ani = FuncAnimation(
        fig,
        update,
        frames=len(x),
        init_func=init,
        blit=True,
        interval=DT * 1e3,
        repeat=False,
    )
    plt.show()


def main():
    print("-------------------")
    print("Simulador quadrotor")
    print("-------------------")

    # Actuation signals
    thrust = np.ones(DATA_L) * MASS * G_CONSTANT
    torque = np.zeros(DATA_L)

    # translational varibles
    a = np.zeros((2, DATA_L))
    v = np.zeros((2, DATA_L))
    p = np.zeros((2, DATA_L))

    # angular variables. Initialized in zero
    theta = np.zeros(DATA_L)
    thetad = np.zeros(DATA_L)
    thetadd = np.zeros(DATA_L)

    # sensores
    accel = np.zeros((2, DATA_L))
    accel_gt = np.zeros((2, DATA_L))
    gyro = np.zeros(DATA_L)
    gps = np.zeros((2, DATA_L))
    vision = np.zeros((2, DATA_L))
    # add optical flow
    op_flow = np.zeros(DATA_L)

    # Setpoints
    yd_ref = np.zeros(DATA_L)
    theta_ref = np.zeros(DATA_L)
    yd_ref[: int(DATA_L * 0.25)] = 2
    theta_ref[int(DATA_L * 0.70) : int(DATA_L * 0.85)] = np.pi / 6
    theta_ref[int(DATA_L * 0.85) :] = -np.pi / 6

    t = np.array(list(range(DATA_L))) * DT

    # Simulate 2 newton law
    # Simular despegue y avance dibujando el suelo
    for i in range(1, DATA_L):  # Pass states are needed, so we start at second
        # Control actuators
        thrust[i], torque[i] = control_actuators(
            theta[i - 1], thetad[i - 1], theta_ref[i], yd_ref[i] - v[1, i - 1]
        )

        # Rotational dynamics
        thetadd[i] = torque[i] / INERTIA
        thetad[i] = (
            thetad[i - 1] + DT * thetadd[i]
        )  # TODO: test trapezoidal integration
        theta[i] = theta[i - 1] + DT * thetad[i]

        # Rotation matrix. Transform body coodinates to inertial coordinates
        c = np.cos(theta[i])
        s = np.sin(theta[i])
        rot_mat = np.array([[c, -s], [s, c]])

        # Translational dynamics
        thrust_rot = rot_mat @ np.array([0, thrust[i]])
        gravity_force = np.array([0, -G_CONSTANT]) * MASS
        a[:, i] = (thrust_rot + gravity_force) / MASS
        v[:, i] = v[:, i - 1] + DT * a[:, i]
        p[:, i] = p[:, i - 1] + DT * v[:, i]

        # simulate sensors
        accel_gt[:, i] = np.linalg.inv(rot_mat) @ a[:, i]
        accel[:, i] = (
            accel_gt[:, i] + randn(2) * ACCEL_NOISE
        )  # TODO: Habría que multiplicarlo por la inversa de rot_mat?
        gyro[i] = thetad[i] + randn(1) * GYRO_NOISE
        if i > GPS_DELAY / DT:
            gps[:, i] = p[:, int(i - GPS_DELAY / DT)] + randn(2) * GPS_NOISE
        else:
            gps[:, i] = None
        vision[:, i] = p[:, i] + randn(2) * VISION_NOISE

    ### Estimación de los estados ###
    # States at delayed time horizon
    v_est = np.empty((2, DATA_L))*np.nan
    p_est = np.empty((2, DATA_L))*np.nan
    theta_est = np.empty(DATA_L)*np.nan

    # States at current time horizon
    v_est_curr = np.empty((2, DATA_L))*np.nan
    p_est_curr = np.empty((2, DATA_L))*np.nan
    theta_est_curr = np.empty(DATA_L)*np.nan

    # Matriz de covarianzas
    P_est = np.empty((5, 5, DATA_L))*np.nan 

    # Buffer de medidas
    max_delay = max(GPS_DELAY, 0)
    buffer_size = int(max_delay / DT)  # TODO: redondear hacia arriba?
    buffer_ekf = [{}] * buffer_size
    gps_insert_pos = int(GPS_DELAY / DT) - 1

    # Buffer de estados
    buffer_output = [{}] * buffer_size

    for i in range(1, DATA_L):

        ## Fill buffer
        # Sensors with no delay (pos 0)
        buffer_ekf.insert(0, {"accel": accel[:, i], "gyro": gyro[i]})
        # gps
        buffer_ekf[gps_insert_pos]["gps"] = gps[:, i]

        ## Pop buffer
        delayed_meas = buffer_ekf.pop()
        accel_delayed = (
            delayed_meas["accel"] if "accel" in delayed_meas.keys() else None
        )
        gyro_delayed = delayed_meas["gyro"] if "gyro" in delayed_meas.keys() else None
        gps_delayed = delayed_meas["gps"] if "gps" in delayed_meas.keys() else None

        [p_est[:, i], v_est[:, i], theta_est[i], P_est[:, :, i]] = (
            ekf_estimator(
                p_est[:, i - 1],
                v_est[:, i - 1],
                theta_est[i - 1],
                P_est[:, :, i - 1],
                accel[:, i],
                gyro[i],
                gps=gps[:, i],
            )
            if not HANDLE_DELAYS
            else ekf_estimator(
                p_est[:, i - 1],
                v_est[:, i - 1],
                theta_est[i - 1],
                P_est[:, :, i - 1],
                accel_delayed,
                gyro_delayed,
                gps=gps_delayed,
            )
        )

        ## Output filter
        [p_est_curr[:, i], v_est_curr[:, i], theta_est_curr[i]] = output_filter(
                p_est_curr[:, i - 1],
                v_est_curr[:, i - 1],
                theta_est_curr[i - 1],
                accel[:, i],
                gyro[i],
            )
        # Fill output buffer
        buffer_output.insert(0, {"p": p_est_curr[:,i], "v":v_est_curr[:, i] , "theta": theta_est_curr[i] })

        ## Corrección

        # Pop buffer
        delayed_state = buffer_output.pop()

        if "p" in delayed_meas.keys():
            p_delayed = delayed_state["p"]
            v_delayed = delayed_state["v"]
            theta_delayed = delayed_state["theta"]

            p_error =       p_est[:, i] - p_delayed
            v_error =       v_est[:, i] - v_delayed
            theta_error =   theta_est[:, i] - theta_delayed
    
            for index, elem  in enumerate(buffer_output):
                # TODO: improve control
                buffer_output[index]["p"] = buffer_output[index]["p"] + theta_error* TAU_P
                buffer_output[index]["v"] = buffer_output[index]["v"] + theta_error* TAU_V
                buffer_output[index]["theta"] = buffer_output[index]["theta"] + theta_error* TAU_THETA

            p_est_curr[:, i] = buffer_output[0]["p"] 
            v_est_curr[:, i] = buffer_output[0]["v"]
            theta_est_curr[i] = buffer_output[0]["theta"]

    # create result folders
    subdir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = IMAGE_FOLDER + subdir_name + os.sep
    os.makedirs(results_path)

    # save parameters
    with open(results_path + "parametros.txt", "w") as f:
        f.write(str(globals()))

    # Plot results
    fig, ax = plt.subplots()
    ax.set_title("X position versus time")
    ax.plot(t, p[0, :], label="P groundtruth")
    if DRAW_ESTIMATED:
        ax.plot(t, p_est[0, :], label="P estimated")
        if SHOW_OUTPUT_FILT:
            ax.plot(t, p_est_curr[0, :], label="P estimated current horizon")
    plt.xlabel("t (s)")
    plt.ylabel("$P_x$ (m)")
    ax.legend()
    plt.savefig(results_path + "x_t." + IMAGE_EXTENSION)

    fig, ax = plt.subplots()
    ax.set_title("Y position versus time")
    ax.plot(t, p[1, :], label="P groundtruth")
    if DRAW_ESTIMATED:
        ax.plot(t, p_est[1, :], label="P estimated")
        if SHOW_OUTPUT_FILT:
            ax.plot(t, p_est_curr[1, :], label="P estimated current horizon")
    plt.xlabel("t (s)")
    plt.ylabel("$P_y$ (m)")
    ax.legend()
    plt.savefig(results_path + "y_t." + IMAGE_EXTENSION)

    fig, ax = plt.subplots()
    ax.set_title("Velocity versus time")
    ax.plot(t, v[0, :], color="tab:red", label="$V_x$ groundtruth", linestyle="--")
    ax.plot(t, v[1, :], color="tab:blue", label="$V_y$ groundtruth", linestyle="--")
    if DRAW_ESTIMATED:
        ax.plot(t, v_est[0, :], color="tab:red", label="$V_x$ estimated")
        ax.plot(t, v_est[1, :], color="tab:blue", label="$V_y$ estimated")
        if SHOW_OUTPUT_FILT:
            ax.plot(t, v_est_curr[0, :], color="tab:red", label="$V_x$ estimated current horizon", linestyle="-.")
            ax.plot(t, v_est_curr[1, :], color="tab:blue", label="$V_y$ estimated current horizon", linestyle="-.")
    plt.xlabel("t (s)")
    plt.ylabel("$V$ (m/s)")
    ax.legend()
    plt.savefig(results_path + "V." + IMAGE_EXTENSION)

    fig, ax = plt.subplots()
    ax.set_title("Tilt versus time")
    ax.plot(t, theta, label="groundtruth")
    if DRAW_ESTIMATED:
        ax.plot(t, theta_est, label="estimated")
        if SHOW_OUTPUT_FILT:
            ax.plot(t, theta_est_curr, label="estimated current horizon")
    plt.xlabel("t (s)")
    plt.ylabel(r"$\theta$ (rad)")
    ax.legend()
    plt.savefig(results_path + "theta." + IMAGE_EXTENSION)

    # Errors
    fig, ax = plt.subplots()
    ax.set_title("Velocity error")
    ax.plot(t, abs(v[0, :]-v_est[0, :]), color="tab:red", label="error $V_x$")
    ax.plot(t, abs(v[1, :]-v_est[1, :]), color="tab:blue", label="error $V_y$")
    ax.plot(t, np.sqrt(P_est[2, 2, :]), color="tab:red", label=r"$\sigma$ estimated $V_x$",linestyle="--")
    ax.plot(t, np.sqrt(P_est[3, 3, :]), color="tab:blue", label=r"$\sigma$ estimated $V_y$",linestyle="-.")
    plt.xlabel("t (s)")
    plt.ylabel("error (m/s)")
    ax.legend()
    plt.savefig(results_path + "V_error." + IMAGE_EXTENSION)

    fig, ax = plt.subplots()
    ax.set_title("Position error")
    ax.plot(t, abs(p[0, :]-p_est[0, :]), color="tab:red", label="error $P_x$")
    ax.plot(t, abs(p[1, :]-p_est[1, :]), color="tab:blue", label="error $P_y$")
    ax.plot(t, np.sqrt(P_est[0, 0, :]), color="tab:red", label=r"$\sigma$ estimated $P_x$",linestyle="--")
    ax.plot(t, np.sqrt(P_est[1, 1, :]), color="tab:blue", label=r"$\sigma$ estimated $P_y$",linestyle="-.")
    plt.xlabel("t (s)")
    plt.ylabel("error (m)")
    ax.legend()
    plt.savefig(results_path + "P_error." + IMAGE_EXTENSION)
    
    fig, ax = plt.subplots()
    ax.set_title("Tilt error")
    ax.plot(t, abs(theta-theta_est), label=r"error $\theta$")
    ax.plot(t, np.sqrt(P_est[4, 4, :]), label=r"$\sigma$ estimated $\theta$")
    plt.xlabel("t (s)")
    plt.ylabel("error (rad)")
    ax.legend()
    plt.savefig(results_path + "theta_error." + IMAGE_EXTENSION)
    

    fig, ax = plt.subplots()
    ax.set_title("Y versus X")
    ax.plot(p[0, :], p[1, :], label="groundtruth")
    if DRAW_ESTIMATED:
        ax.plot(p_est[0, :], p_est[1, :], label="P estimated")
        if SHOW_OUTPUT_FILT:
            ax.plot(p_est_curr[0, :], p_est_curr[1, :], label="P estimated current horizon")
    plt.xlabel("$P_x$ (m)")
    plt.ylabel("$P_y$ (m)")
    ax.legend()
    ax.set_aspect("equal")
    plt.savefig(results_path + "tray." + IMAGE_EXTENSION)

    # Sensors
    fig, ax = plt.subplots()
    ax.set_title("Acelerometer")
    ax.plot(
        t, accel_gt[0, :], color="tab:red", label="$a_x$ groundtruth", linestyle="--"
    )
    ax.plot(
        t, accel_gt[1, :], color="tab:blue", label="$a_y$ groundtruth", linestyle="--"
    )
    ax.plot(t, accel[0, :], color="tab:red", label="$a_x$ measure")
    ax.plot(t, accel[1, :], color="tab:blue", label="$a_y$ measure")
    plt.xlabel("t (s)")
    plt.ylabel("a (m/s)")
    ax.legend()
    plt.savefig(results_path + "accel." + IMAGE_EXTENSION)

    fig, ax = plt.subplots()
    ax.set_title(r"Gyro ($\omega$)")
    ax.plot(t, thetad, label="groundtruth")
    ax.plot(t, gyro, label="measure")
    plt.xlabel("t (s)")
    plt.ylabel(r"$\omega$ (rad/s)")
    ax.legend()
    plt.savefig(results_path + "gyro." + IMAGE_EXTENSION)

    fig, ax = plt.subplots()
    ax.set_title("GPS")
    ax.plot(p[0, :], p[1, :], label="groundtruth")
    ax.plot(gps[0, :], gps[1, :], label="measure", linestyle=" ", marker="x")
    plt.xlabel("$P_x$ (m)")
    plt.ylabel("$P_y$ (m)")
    ax.legend()
    ax.set_aspect("equal")
    plt.savefig(results_path + "gps." + IMAGE_EXTENSION)


    fig, ax = plt.subplots()
    ax.set_title("Matriz de covarianzas")
    # Es diagonal
    ax.plot(t, P_est[0, 1, :], label="$P_{est}$[0,1]")
    ax.plot(t, P_est[0, 2, :], label="$P_{est}$[0,2]")
    ax.plot(t, P_est[0, 3, :], label="$P_{est}$[0,3]")
    ax.plot(t, P_est[0, 4, :], label="$P_{est}$[0,4]")
    ax.plot(t, P_est[1, 2, :], label="$P_{est}$[1,2]")
    ax.plot(t, P_est[1, 3, :], label="$P_{est}$[1,3]")
    ax.plot(t, P_est[1, 4, :], label="$P_{est}$[1,4]")
    ax.plot(t, P_est[2, 3, :], label="$P_{est}$[2,3]")
    ax.plot(t, P_est[2, 4, :], label="$P_{est}$[2,4]")
    ax.plot(t, P_est[3, 4, :], label="$P_{est}$[3,4]")
    ax.plot(t, P_est[0, 0, :], label="$P_x$", linestyle="--")
    ax.plot(t, P_est[1, 1, :], label="$P_y$", linestyle="--")
    ax.plot(t, P_est[2, 2, :], label="$V_x$", linestyle="--")
    ax.plot(t, P_est[3, 3, :], label="$V_y$", linestyle="--")
    plt.xlabel("$t$ (s)")
    ax.legend()
    plt.savefig(results_path + "P_est." + IMAGE_EXTENSION)

    if SHOW_PLOTS: 
        plt.show()

    if SHOW_ANIMATED:
        draw_animation(p[0, :], p[1, :], theta)


if __name__ == "__main__":
    main()
