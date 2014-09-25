package com.neuralnetwork.cnn;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

/**
 * Not to be confused with {@link com.neuralnetwork.nn.layer.NNLayer}
 * The equivalent to this class is a {@link com.neuralnetwork.core.interfaces.ICnnMap}
 *
 * A single CNNLayer is composed of several {@link com.neuralnetwork.core.interfaces.ICnnMap}
 *
 */
class CNNLayer {

    List<MapInfo> lMaps = new LinkedList<MapInfo>();


    public HashSet<MapInfo> getInputMaps()
    {
        HashSet<MapInfo> hsInputMaps = new HashSet<MapInfo>();

        for(MapInfo mapInfo:lMaps)
            hsInputMaps.addAll(mapInfo.getInputMapInfo());

        return hsInputMaps;
    }
}
