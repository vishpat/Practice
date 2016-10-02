(ns project-euler.problem12)

(defn triangle-number
  [n]
  (reduce + n (range n))
  )
