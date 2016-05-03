#lang scheme
(define (max-two x y) (if (> x y) x y))
(define (min-three x y z) (cond ((and (< x y) (< x z)) x)
                          ((and (< y x) (< y z)) y)
                          (else z)
                    )
)

; Exercise 1.3
(define (square-large-two x y z) (- (+ (* x x) (* y y) (* z z)) 
                                    (* (min-three x y z) (min-three x y z))
                                 )
)

; 1.1.7
(define (abs x) ( if (< x 0) (* -1 x) x))
(define (square x) (* x x))
(define (average x y) (/ (+ x y) 2));
(define (improve-guess guess x) (average guess (/ x guess)))
(define (good-enough guess x) (< (abs (- (square guess) x)) 0.001))
(define (good-enough-2 guess x) (< (abs (- (improve-guess guess x) guess)) 0.001))
(define (sqrt-iter guess x) 
            (if (good-enough-2 guess x) 
                guess
                (sqrt-iter (improve-guess guess x) x)
            )
)
(define (sqrt x) (sqrt-iter 1.0 x))

; 1.8
(define (cube x) (* x x x))
(define (good-enough-cube-root guess x) (< (abs (- (cube guess) x)) 0.001))
(define (improve-cube-root-guess guess x) (/ (+ (/ x (square guess)) (* 2 guess)) 3))
(define (cube-root-iter guess x) 
            (if (good-enough-cube-root guess x) 
                guess
                (cube-root-iter (improve-cube-root-guess guess x) x)
            )
)
(define (cube-root x) (cube-root-iter 1.0 x))

; Factorial recurisve
(define (factorial-recursive x) (if (<= x 1) 1 (* x (factorial-recursive (- x 1)))))

; Factorial iterative
  
(define (factorial-iterative x)
  (define (factorial-iter x counter product)
    (if (< counter x) (factorial-iter x (+ counter 1) (* counter product)) (* counter product))
  )
  (factorial-iter x 1 1)
)

;
