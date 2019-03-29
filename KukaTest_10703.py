from KukaEnv_HER import KukaVariedObjectEnv
import time, sys

def main(argv):
    environment = KukaVariedObjectEnv(argv[0], renders=True,isDiscrete=False, maxSteps = 10000000)
    motorsIds = []
    dv = 0.01
    motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("yaw",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,0.3))

    while True:
        state = environment.reset()
        done = False
        print('sample action')
        print(environment.action_space.sample())
        while (not done):
            action=[]
            for motorId in motorsIds:
                action.append(environment._p.readUserDebugParameter(motorId))
            print('real action')
            print(action)
            state, reward, done, info = environment.step(action)
            print(environment.get_feature_vec_observation())

if __name__=="__main__":
    if len(sys.argv) < 2:
        print('Must call with file path to "items" directory')
        exit()
    main(sys.argv[1:])
