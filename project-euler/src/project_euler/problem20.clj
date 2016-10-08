(ns project-euler.problem20 
  (:require [project-euler.util :as util]))

(defn solve
  []
  (reduce + (map #(Integer/parseInt %) (map str (seq (util/str-num-factorial "100")))))
  )
