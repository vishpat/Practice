(define (max-two x y) (if (> x y) x y))
(define (min-three x y z) (cond ((and (< x y) (< x z)) x)
                          ((and (< y x) (< y z)) y)
                          (else z)
                    )
)