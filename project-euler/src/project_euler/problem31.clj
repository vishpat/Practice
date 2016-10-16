(ns project-euler.problem31)

(def coins '(1 2 5 10 20 50 100 200))
(def combinations (atom {0 0}))

(defn init-combinations
  []
  (loop [amount 1]
    (when (<= amount 200)
          (swap! combinations assoc amount (reduce + (map #(if (= 0 (mod amount %)) 1 0) coins))) 
          (recur (inc amount)))
   )
)

(defn solve
  []
  (do
    (init-combinations)
    (println @combinations)
    (loop [amount 0]
      (if (= amount 201) (get @combinations 200)
        (do
          (let [amount-combinations (get @combinations amount)]
            (swap! combinations assoc amount 
                (+ amount-combinations 
                (reduce + (map #(let [value (get @combinations (- amount %))] (if (nil? value) 0 value)) coins))))
          )
          (println amount " " (get @combinations amount))
          (recur (inc amount)))
        )
      )
    )
  )
