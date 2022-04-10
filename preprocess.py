def process_video():
    # https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/#:~:text=The%20first%20step%20towards%20reading%20a%20video%20file,camera%20by%20passing%20%E2%80%982%E2%80%99%20and%20so%20on.%20Python?msclkid=e7282deeb49611eca5009d56db360f21
    import cv2
    import numpy as np
    from includes import Utils

    vc = cv2.VideoCapture('Chaplin_512kb.mp4')
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    if not vc.isOpened():
        Utils.color_print('could not open the video', type_='error')

    frames = np.zeros((num_frames,240,320,3), dtype=np.uint8)
    counter = 0
    # frames = np.zeros((num_frames))
    while vc.isOpened():
        if counter%1000 == 0:
            print('{counter} frames done.'.format(counter=counter))
        ret, frame = vc.read()
        if ret == False:
            break
        frames[counter, :, :, :] = frame

        # cv2.imshow('Frame {i}'.format(i=0), frame)

        # cv2.waitKey()

        # if cv2.waitKey(25) & 0xff == ord('q'):
        #     break

        counter += 1

    vc.release()
    # cv2.destroyAllWindows()

    print('Writing {N} frames in Chaplin_512kb.npy'.format(N=counter))
    np.save('Chaplin_512kb.npy', frames)

def process_audio():
    import librosa
    import numpy as np

    input_, sr_ = librosa.load('Chaplin_512kb_L.wav', sr=16000, mono=True)
    print('Read audio: ', np.shape(input_), sr_)

    n_fft = 320

    stft_ = librosa.stft(input_, n_fft=n_fft, hop_length=160, win_length=320, window='hann', center=True, dtype=None, pad_mode='constant')
    print('Extracted stft: ', np.shape(stft_))
    stft_cat = np.zeros((n_fft, np.shape(stft_)[1]))
    stft_cat[0:160] = np.real(stft_[:-1,:])
    stft_cat[160:] = np.imag(stft_[:-1,:])

    print('Writing {N} complex frames in Chaplin_512kb_stft.npy'.format(N=np.shape(stft_cat)[1]))
    np.save('Chaplin_512kb_stft.npy', stft_cat)




if __name__ == '__main__':
    process_audio()