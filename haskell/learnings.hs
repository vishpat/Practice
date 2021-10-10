import Data.Char

doubleMe x = 2*x
addUs x y = x + y
doubleSmallNumber x = if x > 100 then x else x*2

-- pi :: Float
pi = 3.14

areaOfACircle :: Float -> Float 
areaOfACircle r = 3.14 *r*r 

convertUppercase :: [Char] -> [Char]
convertUppercase xs = [toUpper x | x <- xs]

-- String to dataype
read "6" :: Int
read "7.0" :: Float

-- Type casting
20 :: Float
