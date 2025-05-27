def perform_operations(self):
    # Perform operations using the S-bit memory bridge
    index1 = 0
    index2 = 1
    and_result_index = self.memory_bridge.and_op(index1, index2)
    or_result_index = self.memory_bridge.or_op(index1, index2)
    not_result_index = self.memory_bridge.not_op(index1)

    # Measure the results
    and_result = self.memory_bridge.measure_sbit(and_result_index)
    or_result = self.memory_bridge.measure_sbit(or_result_index)
    not_result = self.memory_bridge.measure_sbit(not_result_index)

    return and_result, or_result, not_result
