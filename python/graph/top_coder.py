# SRM 150 RoboCourier

2r = 5 
kx = 4 
ky = 3 

class RoboCourier:
   
    

    def timeToDeliver(self, path_str):
        
        mapping = {
             0  : (2r, 0),
            60  : (kx, ky),
           120  : (kx, -1*ky),
           180  : (-2r, 0),
           240  : (-1*kx, -1*ky),
           300  : (-1*kx, ky)
        }

        path = list(path_str)
        
        cur_state = (0, 0)
        cur_angle = 0

        for instruction in path:
            
            cur_x, cur_y = cur_state

            if instruction == 'F':
                x, y = mapping[cur_angle]
                next_state = (cur_x + x, cur_y + y)
            elif instruction == 'R':
                cur_angle += 60
                cur_angle %= 360
            elif instruction == 'L':
                cur_angle -= 60
                cur_angle %= 360
                

