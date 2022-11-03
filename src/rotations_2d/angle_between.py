import numpy as np
import matplotlib.pyplot as plt

def fix_to_range(angle):
    if angle>np.pi:
        angle-=2*np.pi
    if angle<=-np.pi:
        angle+=2*np.pi
    return angle

def angle_between(v1, v2):
    return fix_to_range(np.arctan2(v2[1], v2[0])-np.arctan2(v1[1],v1[0]))


def test_90():
    vec1 = np.array([1, 0])
    vec2 = np.array([0, 1])
    res = angle_between(vec1, vec2)
    assert(np.abs(res-np.pi/2)<1e-4), res

def test_min90():
    vec1 = np.array([0, 1])
    vec2 = np.array([1, 0])
    res = angle_between(vec1, vec2)
    assert(np.abs(res-(-np.pi/2))<1e-4), res
def test_180():
    vec1 = np.array([1, 0])
    vec2 = np.array([-1, 0])
    res = angle_between(vec1, vec2)
    assert(np.abs(res-(np.pi))<1e-4), res

def test_270():
    vec1 = np.array([1, 0])
    vec2 = np.array([0, -1])
    res = angle_between(vec1, vec2)
    assert(np.abs(res-(-np.pi/2))<1e-4), res


def get_rotation_matrix(th):
    return np.array([[np.cos(th), -np.sin(th)], 
                    [np.sin(th), np.cos(th)]])

def test_rotation_matrix():
    vec0 = np.array([1, 0])
    vec1 = get_rotation_matrix(np.pi/2)@vec0
    exp_vec1 = np.array([0, 1])
    assert(np.linalg.norm(vec1-exp_vec1)<1e-5), vec1

def test_rotate_and_retrieve():
    vec0 = np.array([1, 0])

    N = 100
    for _ in range(N):
        th = np.random.uniform(-np.pi, np.pi)
        vec1 = get_rotation_matrix(th)@vec0

        angle_calc = angle_between(vec0, vec1)
        assert(np.abs(angle_calc-th)<1e-5), angle_calc


test_90()
test_min90()
test_180()
test_270()
test_rotation_matrix()
test_rotate_and_retrieve()