# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:16:09 2022

@author: bart1
"""

import tqdm
import cv2

def show_anim(folder, n_frames, framedur = 50, title = 'animation',
              write_name = None, resolution = None):
    
    frames = []
    for i in tqdm.tqdm(range(n_frames)):
        frame = cv2.imread(f"{folder}/{i}.png")
        
        if resolution:
            frame = cv2.resize(frame,resolution)
        
        frames.append(frame)
        
    i = 0
    
    while True:
        cv2.imshow(title, frames[i % n_frames])
        key = cv2.waitKey(framedur)
        
        if key == ord('q'):
            break
        
        i += 1
        
    cv2.destroyAllWindows()
    
    if write_name:
        
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out    = cv2.VideoWriter(f"{folder}/{write_name}.mp4", fourcc, 1000/framedur,
                                  (frames[0].shape[1],frames[0].shape[0]))
        
        for frame in tqdm.tqdm(frames):
            out.write(frame)
            
        out.release()

if __name__ == "__main__":
        
    show_anim(r"Results\GOL_Menger", 1000, write_name = "GOL_Menger", resolution = (1000,1000), framedur = 50)