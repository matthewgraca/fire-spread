"""
Takes in a Dataset

For each hour h:

    Gather the confidences from hour [1, h]
    Calculate the cumulative max of the confidences from [1, h]
    Apply spatial smoothing
    Calculate the max confidence of the entire frame s_h; the scaling factor (+ exceptions like t >= 500)
    For the entire frame, divide each pixel by s_h

"""
