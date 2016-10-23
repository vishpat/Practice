(ns project-euler.problem31)

(def coins '(1 2 5 10 20 50 100 200))
(def combinations (atom {1 1}))
(def final-amount 100) 

(defn solve
  []
  (do
    (doseq [coin coins]
      (swap! combinations assoc coin 1))
    (println @combinations)
    (loop [amount 2]
      (if (= amount (inc final-amount)) (get @combinations final-amount)
        (do
            (swap! combinations assoc amount 
                (reduce + (map #(let [value (get @combinations (- amount %))] (if (nil? value) 0 value)) coins)))
          (println amount " " (get @combinations amount))
          (recur (inc amount)))
        )
      )
    )
  )
