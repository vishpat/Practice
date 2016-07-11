(ns project-euler.core)

(defn square-numbers
  [[& numbers]]
  (loop [[n0 & nums] numbers squares []]
    (if n0
      (do
          (def n2 (* n0 n0))
          (recur nums (conj squares n2)))
      squares
      )))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println (square-numbers [1 2 3 4])))

