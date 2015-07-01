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
