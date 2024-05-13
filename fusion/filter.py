import numpy as np


#################### Top Level Functions ####################
def estimate(t, w, a, q0):
    N = len(t)
    q = [None] * N
    bx, by, bz = [None] * N, [None] * N, [None] * N
    bx_err, by_err, bz_err = [None] * N, [None] * N, [None] * N
    ekf = EKF(q0=q0)

    q[0] = ekf.x[0:4]
    bx[0], by[0], bz[0] = ekf.x.flat[4:7]
    bx_err[0] = np.sqrt(ekf.P[4, 4])
    by_err[0] = np.sqrt(ekf.P[5, 5])
    bz_err[0] = np.sqrt(ekf.P[6, 6])
    for i in range(1, N):
        dt = (t[i] - t[i-1]) / 1000.0
        wi = [w[0][i], w[1][i], w[2][i]]
        ai = [a[0][i], a[1][i], a[2][i]]
        ekf.predict(wi, dt)
        ekf.update(ai)
        
        q[i] = ekf.x[0:4]
        bx[i], by[i], bz[i] = ekf.x.flat[4:7]
        bx_err[i] = np.sqrt(ekf.P[4, 4])
        by_err[i] = np.sqrt(ekf.P[5, 5])
        bz_err[i] = np.sqrt(ekf.P[6, 6])

    return q, [bx, by, bz], [bx_err, by_err, bz_err]

def quaternion_to_euler_degrees(q):
    N = len(q)
    roll = [None] * N
    pitch = [None] * N
    yaw = [None] * N

    for i in range(N):
        if q[i] is not None:
            r, p, y = quaternion_to_euler(q[i])
            roll[i] = np.degrees(r)
            pitch[i] = np.degrees(p)
            yaw[i] = np.degrees(y)
    
    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    c_roll   = np.cos(roll / 2)
    s_roll   = np.sin(roll / 2)
    c_pitch = np.cos(pitch / 2)
    s_pitch = np.sin(pitch / 2)
    c_yaw   = np.cos(yaw / 2)
    s_yaw   = np.sin(yaw / 2)

    qw = c_roll * c_pitch * c_yaw + s_roll * s_pitch * s_yaw
    qx = s_roll * c_pitch * c_yaw - c_roll * s_pitch * s_yaw
    qy = c_roll * s_pitch * c_yaw + s_roll * c_pitch * s_yaw
    qz = c_roll * c_pitch * s_yaw - s_roll * s_pitch * c_yaw
    return [qw, qx, qy, qz]

def quaternion_to_euler(q):
    qw, qx, qy, qz = q.flat
    roll  = np.arctan2(2 * (qw * qx + qy * qz),
                       1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))
    yaw   = np.arctan2(2 * (qw * qz + qx * qy),
                       1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw




g = 9.80665


def normalize(v):
    return v / np.linalg.norm(v)


def quaternion_identity():
    return np.c_[[1, 0, 0, 0]]


def quaternion_to_matrix(q_unit):
    w, x, y, z = q_unit.flat
    w2 = w * w
    x2 = x * x
    y2 = y * y
    z2 = z * z
    r11 = w2 + x2 - y2 - z2
    r12 = 2 * (x * y - w * z)
    r13 = 2 * (w * y + x * z)
    r21 = 2 * (x * y + w * z)
    r22 = w2 - x2 + y2 - z2
    r23 = 2 * (y * z - w * x)
    r31 = 2 * (x * z - w * y)
    r32 = 2 * (y * z + w * x)
    r33 = w2 - x2 - y2 + z2
    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])


def quaternion_multiply(p, q):
    pw, px, py, pz = p.flat
    qw, qx, qy, qz = q.flat
    rw = pw * qw - px * qx - py * qy - pz * qz
    rx = pw * qx + px * qw + py * qz - pz * qy
    ry = pw * qy - px * qz + py * qw + pz * qx
    rz = pw * qz + px * qy - py * qx + pz * qw
    return np.c_[[rw, rx, ry, rz]]


def quaternion_from_axis_angle(unit_axis, angle_rad):
    ux, uy, uz = unit_axis.flat
    sin_half = np.sin(angle_rad / 2)
    qw = np.cos(angle_rad / 2)
    qx = ux * sin_half
    qy = uy * sin_half
    qz = uz * sin_half
    return np.c_[[qw, qx, qy, qz]]


def quaternion_from_rotation_vector(v, eps=0):
    angle_rad = np.linalg.norm(v)

    # Guard against division by zero
    if angle_rad > eps:
        unit_axis = v / angle_rad
        q_unit = quaternion_from_axis_angle(unit_axis, angle_rad)
    else:
        q_unit = quaternion_identity()

    return q_unit


def get_F(x, w, dt):
    qw, qx, qy, qz, bx, by, bz = x.flat
    wx, wy, wz = w.flat
    return np.array([
        [             1, dt*(-wx + bx)/2, dt*(-wy + by)/2, dt*(-wz + bz)/2,  dt*qx/2,  dt*qy/2,  dt*qz/2],
        [dt*(wx - bx)/2,               1, dt*( wz - bz)/2, dt*(-wy + by)/2, -dt*qw/2,  dt*qz/2, -dt*qy/2],
        [dt*(wy - by)/2, dt*(-wz + bz)/2,               1, dt*( wx - bx)/2, -dt*qz/2, -dt*qw/2,  dt*qx/2],
        [dt*(wz - bz)/2, dt*( wy - by)/2, dt*(-wx + bx)/2,               1,  dt*qy/2, -dt*qx/2, -dt*qw/2],
        [             0,               0,               0,               0,        1,        0,        0],
        [             0,               0,               0,               0,        0,        1,        0],
        [             0,               0,               0,               0,        0,        0,        1]
    ])


def get_W(x, dt):
    qw, qx, qy, qz = x.flat[0:4]
    return dt/2 * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
        [  0,   0,   0],
        [  0,   0,   0],
        [  0,   0,   0]
    ])


def get_H(x):
   qw, qx, qy, qz = x.flat[0:4]
   return 2 * g * np.array([
        [ qy, -qz,  qw, -qx, 0, 0, 0],
        [-qx, -qw, -qz, -qy, 0, 0, 0],
        [-qw,  qx,  qy, -qz, 0, 0, 0]
   ])


def f(x, w, dt):
    q = x[0:4]
    b = x[4:7]

    d_ang = (w - b) * dt
    dq = quaternion_from_rotation_vector(d_ang)
    q = quaternion_multiply(q, dq)
    q = normalize(q)

    return np.r_[q, b]


def h(x):
    q = x[0:4]
    R_from_body = quaternion_to_matrix(q)
    return R_from_body.T @ np.c_[[0, 0, -g]]


class EKF:
    def __init__(self, q0=[1,0,0,0], b0=[0,0,0], init_gyro_bias_err=0.01,
                 gyro_noise=1.0, gyro_bias_noise=0.0005, accelerometer_noise=1.0):
        '''
        q0 -- initial orientation (unit quaternion) [qw, qx, qy, qz]
        b0 -- initial gyro bias [rad/sec] [bx, by, bz]
        init_gyro_bias_err -- initial gyro bias uncertainty (1 standard deviation) [rad/sec]
        gyro_noise -- Gyro noise (1 standard deviation) [rad/sec]
        gyro_bias_noise -- Gyro bias noise (1 standard deviation) [rad/sec]
        accelerometer_noise -- Accelerometer measurement noise (1 standard deviation) [m/s^2]
        '''
        self.x = np.c_[q0 + b0] # State
        self.P = np.zeros((7, 7))
        self.P[0:4, 0:4] = np.identity(4) * 0.01
        self.P[4:7, 4:7] = np.identity(3) * (init_gyro_bias_err ** 2)
        self.Q = np.identity(3) * (gyro_noise ** 2)
        self.Q_bias = np.zeros((7, 7))
        self.Q_bias[4:7, 4:7] = np.identity(3) * (gyro_bias_noise ** 2)
        self.R = np.identity(3) * (accelerometer_noise ** 2)
    
    def predict(self, w, dt):
        '''
        w -- gyro measurement in front-right-down coordinates [rad/sec]
        dt -- [sec]
        '''
        w = np.c_[w]
        F = get_F(self.x, w, dt)
        W = get_W(self.x, dt)
        self.x = f(self.x, w, dt)
        self.P = F @ self.P @ F.T + W @ self.Q @ W.T + self.Q_bias

    def update(self, a):
        '''
        a -- accelerometer measurement in front-right-down coordinates [m/s^2]
        '''
        a = np.c_[a]
        z = g * normalize(a)
        y = z - h(self.x)
        H = get_H(self.x)
        S = H @ self.P @ H.T + self.R
        K = (self.P @ H.T) @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.x[0:4] = normalize(self.x[0:4])
        self.P = (np.identity(7) - K @ H) @ self.P