(ns project-euler.problem15)

(def size 21)
(def weight-matrix (atom {'(0 0) 1}))

(defn init-map
  []
  (doseq [x (take size (range)) y (take size (range))]
    (swap! weight-matrix assoc (list x y) 0)
   )
  (swap! weight-matrix assoc (list 0 0) 1)
)

(defn solve
  []
  (do
   (init-map)
   (doseq [x (take size (range)) y (take size (range))]
       (let [my-weight (get @weight-matrix (list x y)) next-x (inc x) next-y (inc y) ]
          (when (< next-x size) (swap! weight-matrix assoc (list next-x y) (+ my-weight (get @weight-matrix (list next-x y)))))
          (when (< next-y size) (swap! weight-matrix assoc (list x next-y) (+ my-weight (get @weight-matrix (list x next-y)))))
         )
       )
     )
  (get @weight-matrix (list (- size 1) (- size 1)))
  )
