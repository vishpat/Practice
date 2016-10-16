(ns project-euler.problem31)

(def coins '(1 2 5 10 20 50 100 200))
(def combinations (atom {0 0}))
(def final-amount 4) 

(defn solve
  []
  (do
    (doseq [coin coins]
      (swap! combinations assoc coin 1))
    (println @combinations)
    (loop [amount 0]
      (if (= amount (inc final-amount)) (get @combinations final-amount)
        (do
          (let [x (get @combinations amount) amount-combinations (if x x 0)]
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
