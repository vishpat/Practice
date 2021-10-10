import Data.Char
import System.Directory.Internal.Prelude (Num, Integral)

doubleMe x = 2*x
addUs x y = x + y
doubleSmallNumber x = if x > 100 then x else x*2

-- pi :: Float
pi = 3.14

areaOfACircle :: Float -> Float 
areaOfACircle r = 3.14 *r*r 

convertUppercase :: [Char] -> [Char]
convertUppercase xs = [toUpper x | x <- xs]

length' :: (Num b) => [a] -> b 
length' xs = sum [1 | x <- xs ]

-- pattern matching
factorial :: (Integral a) => a -> a 
factorial 0 = 1
factorial x = x * factorial(x - 1)

first :: (a, b, c) -> a
first (x, _, _) = x 

second :: (a, b, c) -> b 
second (_, y, _) = y 

third :: (a, b, c) -> c
third (_,_, z) = z 

head' :: [a] -> a
head' [] = error "Empty List"
head' (x:xs) = x 

length2 :: (Num b) => [a] -> b
length2 [] = 0
length2 (x:xs) = 1 + length2 xs
