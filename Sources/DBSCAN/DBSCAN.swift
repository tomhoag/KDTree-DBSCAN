/**
 A density-based, non-parametric clustering algorithm
 ([DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)).

 Given a set of points in some space,
 this algorithm groups points with many nearby neighbors
 and marks points in low-density regions as outliers.

 - Authors: Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Xu, Xiaowei (1996)
            "A density-based algorithm for discovering clusters
            in large spatial databases with noise."
            _Proceedings of the Second International Conference on
            Knowledge Discovery and Data Mining (KDD-96)_.
 */

import KDTree
import SwiftUI


public struct DBSCAN<Value: Equatable & Hashable & KDTreePoint> {
    private class Point: Equatable {
        typealias Label = Int

        static var dimensions: Int { Value.dimensions }

        let value: Value
        var label: Label?

        init(_ value: Value) {
            self.value = value
        }

        static func == (lhs: Point, rhs: Point) -> Bool {
            return lhs.value == rhs.value
        }

        func kdDimension(_ dimension:Int) -> Double {
            return value.kdDimension(dimension)
        }

        func squaredDistance(to otherPoint: DBSCAN<Value>.Point) -> Double {
            var sum = 0.0
            for d in 0..<Self.dimensions {
                let dx = kdDimension(d) - otherPoint.kdDimension(d)
                sum += dx * dx
            }
            return sum
        }
    }

    /// The values to be clustered.
    public var values: [Value]
    public var kdtree: KDTree<Value>?

    /// Creates a new clustering algorithm with the specified values.
    /// - Parameter values: The values to be clustered.
    public init(_ values: [Value], useKDTree: Bool = true) {
        self.values = values
        if useKDTree {
            kdtree = KDTree(values: values)
        }
    }

    /**
     Clusters values according to the specified parameters.

     - Parameters:
       - epsilon: The maximum distance from a specified value
                  for which other values are considered to be neighbors.
       - minimumNumberOfPoints: The minimum number of points
                                required to form a dense region.
       - distanceFunction: A function that computes
                           the distance between two values.
     - Throws: Rethrows any errors produced by `distanceFunction`.
     - Returns: A tuple containing an array of clustered values
                and an array of outlier values.
    */
    public func callAsFunction(epsilon: Double, minimumNumberOfPoints: Int, distanceFunction: (Value, Value) throws -> Double) rethrows -> (clusters: [[Value]], outliers: [Value]) {
        precondition(minimumNumberOfPoints >= 0)

        let n = values.count
        guard n > 0 else { return ([], []) }

        // labels per index (nil = unlabeled)
        var labels = Array<Int?>(repeating: nil, count: n)
        var currentLabel = 0

        for i in 0..<n {
            if labels[i] != nil { continue }

            // find initial neighbors (indices) of i
            var neighbors: [Int] = []
            neighbors.reserveCapacity(16)
            for j in 0..<n {
                if try distanceFunction(values[i], values[j]) < epsilon {
                    neighbors.append(j)
                }
            }

            if neighbors.count >= minimumNumberOfPoints {
                defer { currentLabel += 1 }
                labels[i] = currentLabel

                // index-based queue with head pointer to avoid O(n) removeFirst()
                var queue = neighbors
                var head = 0

                while head < queue.count {
                    let neighborIndex = queue[head]; head += 1

                    if labels[neighborIndex] != nil { continue }

                    labels[neighborIndex] = currentLabel

                    // find neighbors of neighborIndex
                    var n1: [Int] = []
                    n1.reserveCapacity(8)
                    for j in 0..<n {
                        if try distanceFunction(values[neighborIndex], values[j]) < epsilon {
                            n1.append(j)
                        }
                    }

                    if n1.count >= minimumNumberOfPoints {
                        queue.append(contentsOf: n1)
                    }
                }
            }
        }

        // Build clusters and outliers from labels
        var clustersDict: [Int: [Value]] = [:]
        clustersDict.reserveCapacity(currentLabel)
        var outliers: [Value] = []
        outliers.reserveCapacity(n / 10)

        for (idx, v) in values.enumerated() {
            if let lbl = labels[idx] {
                clustersDict[lbl, default: []].append(v)
            } else {
                outliers.append(v)
            }
        }

        let clusters = clustersDict.keys.sorted().map { clustersDict[$0]! }
        return (clusters, outliers)
    }

    public func callAsFunction(epsilon: Double, minimumNumberOfPoints: Int) -> (clusters: [[Value]], outliers: [Value]) {
        precondition(minimumNumberOfPoints >= 0)
        precondition(kdtree != nil, "KDTree must be initialized")

        let tree = kdtree! // local strong reference
        let n = values.count
        guard n > 0 else { return ([], []) }

        // labels per index (nil = unlabeled)
        var labels = Array<Int?>(repeating: nil, count: n)

        // Map Value -> index for O(1) conversions from KDTree results to indices
        let valueToIndex: [Value: Int] = Dictionary(uniqueKeysWithValues: values.enumerated().map { ($1, $0) })

        var currentLabel = 0

        // Work through each point by index
        for i in 0..<n {
            if labels[i] != nil { continue }

            // initial neighbors as indices
            let neighborValues = tree.allPoints(within: epsilon, of: values[i])
            var neighbors = neighborValues.compactMap { valueToIndex[$0] }

            if neighbors.count >= minimumNumberOfPoints {
                defer { currentLabel += 1 }
                labels[i] = currentLabel

                // Use an index-based queue (head pointer) to avoid O(n) removeFirst
                var queue = neighbors
                var head = 0

                while head < queue.count {
                    let neighborIndex = queue[head]
                    head += 1

                    if labels[neighborIndex] != nil { continue }

                    labels[neighborIndex] = currentLabel

                    let n1Values = tree.allPoints(within: epsilon, of: values[neighborIndex])
                    let n1 = n1Values.compactMap { valueToIndex[$0] }

                    if n1.count >= minimumNumberOfPoints {
                        queue.append(contentsOf: n1)
                    }
                }
            }
        }

        // Build clusters and outliers from labels
        var clustersDict: [Int: [Value]] = [:]
        clustersDict.reserveCapacity(currentLabel)
        var outliers: [Value] = []
        outliers.reserveCapacity(n / 10)

        for (idx, v) in values.enumerated() {
            if let lbl = labels[idx] {
                clustersDict[lbl, default: []].append(v)
            } else {
                outliers.append(v)
            }
        }

        // Return clusters in label order
        let clusters = clustersDict.keys.sorted().map { clustersDict[$0]! }
        return (clusters, outliers)
    }
}

