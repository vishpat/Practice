(ns project-euler.problem15)

(def size 20)
(def weight-matrix {'(0 0) 0})

(defn init-map
  []
  (doseq [x (take size (range)) y (take size (range))]
    (def weight-matrix (assoc weight-matrix (list x y) 0))
   )
)

(defn solve
  []
  (do
   (init-map)
   (doseq [x (take size (range)) y (take size (range))]
       (let [my-weight (get weight-matrix (list x y)) next-x (inc x) next-y (inc y) ]
         (println (str (str (get weight-matrix '(next-x y))) (str (get weight-matrix (list x next-y)))))
          (cond 
            (< next-x size) (def weight-matrix (update weight-matrix (list next-x y) inc)) 
            (< next-y size) (def weight-matrix (update weight-matrix (list x next-y) inc)) 
          )
         )
       )
     )
  (get weight-matrix (list (- size 1) (- size 1)))
  )
