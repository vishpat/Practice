(ns project-euler.core
  (require [project-euler.util :as util]
           [project-euler.problem13 :as problem13]
           ))

(defn problem-20
  []
  (reduce + (map #(Integer/parseInt %) (map str (seq (util/str-num-factorial "100")))))
  )
