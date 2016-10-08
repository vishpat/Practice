(ns project-euler.problem20 
  (:require [project-euler.util :as util]))

(defn solve
  []
  (util/calculate-str-num-digit-sum (util/str-num-factorial "100"))
  )
