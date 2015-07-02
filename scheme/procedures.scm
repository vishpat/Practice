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
(define (sqrt-iter guess x) 
            (if (good-enough guess x) 
                guess
                (sqrt-iter (improve-guess guess x) x)
            )
)
(define (sqrt x) (sqrt-iter 1.0 x))
