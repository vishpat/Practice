(ns project-euler.core
  (require [project-euler.util :as util]))

(defn problem-20
  []
  (reduce + (map #(Integer/parseInt %) (map str (seq (util/factorial "100")))))
  )
