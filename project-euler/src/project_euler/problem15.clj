(ns project-euler.problem15)

(def size 20)
(def weight {'(0 0) 1})

(defn solve
  []
  (doseq [x (take size (range)) y (take size (range))]
     (println (str "x = " (str x) ", y = " (str y)))
     )
  (get weight '((- size 1) (- size 1)))
  )
