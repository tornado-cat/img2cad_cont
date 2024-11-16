# Gerekli kütüphaneleri içe aktarıyoruz.
import ezdxf.math  # DXF dosyalarını işlemek ve geometrik hesaplamalar için.
import numpy as np  # Sayısal hesaplamalar ve vektör/matris işlemleri için.
import cv2  # Görüntü işleme için (bu kodda kullanılmamış ama potansiyel kullanım için dahil edilmiş).
import ezdxf  # DXF dosyalarını işlemek ve veri çıkarmak için.
from probreg import cpd  # CPD (Coherent Point Drift) algoritması ile nokta bulutlarını hizalamak için.

# DXF dosyasını nokta bulutuna dönüştüren bir fonksiyon.
def cad_to_pointcloud(filepath, line_sections=100):
    """
    DXF CAD dosyasını okuyarak geometrik şekilleri nokta bulutuna dönüştürür.
    
    filepath: DXF dosyasının yolu.
    line_sections: Her bir çizgi veya yay için oluşturulacak nokta sayısı.
    """
    # DXF dosyasını okuyoruz.
    drawing_file = ezdxf.readfile(filepath)
    # Model uzayını (çizim yapılan alan) alıyoruz.
    drawing_msp = drawing_file.modelspace()

    # Noktaları saklamak için bir liste oluşturuyoruz.
    point_arrays = []
    # Model alanındaki varlıkların türlerini kontrol ediyoruz.
    # Her bir varlığı işleyerek noktaları oluşturacağız.
    
    # DXF dosyası içerisindeki varlıkları döngüyle geziyoruz.
    for entity in drawing_msp:
        # Eğer varlık bir yay (ARC) veya çember (CIRCLE) ise:
        if entity.dxftype() in ['ARC', 'CIRCLE']:
            # Varlığı bir spline (düzgün eğri) haline getiriyoruz.
            sp = entity.to_spline()
            # Eğriyi oluşturmak için gerekli araçları alıyoruz.
            ct = sp.construction_tool()
            # Yayı veya çemberi belirli bir noktalar setine böleriz.
            points = np.array(list(ct.points(np.linspace(0, ct.max_t, line_sections))), dtype=np.float32)
            # Noktaları 2 boyutlu hale getiriyoruz.
            points = np.reshape(points, (-1, 2))
            # Bu noktaları nokta listesine ekliyoruz.
            point_arrays.append(points)

        # Eğer varlık bir LWPOLYLINE (hafif ağırlıklı çoklu çizgi) ise:
        elif entity.dxftype() == 'LWPOLYLINE':
            # Çoklu çizgi üzerindeki noktaları ve eğim parametrelerini alıyoruz.
            point_params = np.array(list(entity.get_points("xyb")))
            temp = []  # Bu çoklu çizgiye ait noktaları geçici olarak saklamak için.

            # Başlangıç noktasını belirliyoruz. Eğer kapalı bir çizimse, başlangıç ayarlanır.
            start_idx = 0
            if entity.is_closed:
                start_idx = -1

            # Çoklu çizginin her iki nokta arasını işleriz.
            for i in range(start_idx, len(point_params) - 1):
                start = point_params[i, :2]  # Başlangıç noktası.
                end = point_params[i + 1, :2]  # Bitiş noktası.
                bulge = point_params[i, 2]  # Eğrilik parametresi.

                # Eğer eğrilik varsa (yay şekli oluşuyorsa):
                if bulge != 0:
                    # Merkezi, başlangıç ve bitiş açısını ve yarıçapı hesaplarız.
                    center, sa, ea, rad = ezdxf.math.bulge_to_arc(start, end, bulge)

                    # Eğer bitiş açısı negatifse, 2π ekleriz.
                    if ea < 0:
                        ea += 2 * np.pi

                    # Açılar arasındaki farkı kontrol ediyoruz.
                    delta = ea - sa
                    if abs(delta) > np.pi:
                        # Açıyı normalize etmek için gerekli düzeltmeyi yapıyoruz.
                        delta -= 2 * np.pi
                        ea = sa + delta

                    # Yeni bir yay varlığı oluşturuyoruz (sadece görselleştirme için).
                    new_ent = drawing_msp.add_arc(center, rad, sa, ea)

                    # Yayı noktalar dizisi olarak hesaplıyoruz.
                    ct = new_ent.construction_tool()
                    v = ct.vertices(np.linspace(np.rad2deg(sa), np.rad2deg(ea), 100))
                    points = np.array(list(v), dtype=np.float32)
                    points = np.reshape(points, (-1, 2))
                    # Noktaları geçici listeye ekliyoruz.
                    temp.append(points)
                else:
                    # Eğer eğrilik yoksa (düz bir çizgi):
                    x = np.linspace(0, 1, line_sections).reshape((line_sections, 1))
                    x = np.repeat(x, 2, 1)  # Noktaların boyutunu ayarlarız.
                    delta = end - start
                    x = x * delta + start
                    temp.append(x)
            # Geçici noktaları birleştirip ana listeye ekliyoruz.
            temp = np.reshape(np.concatenate(temp), (-1, 2))
            point_arrays.append(temp)

        # Eğer varlık düz bir çizgi (LINE) ise:
        elif entity.dxftype() == 'LINE':
            x = np.linspace(0, 1, line_sections).reshape((line_sections, 1))
            x = np.repeat(x, 2, 1)
            start = np.array(entity.dxf.start)[:2]
            end = np.array(entity.dxf.start)[:2]
            delta = end - start
            x = x * delta + start

            point_arrays.append(x)

    # Tüm noktaları birleştiriyoruz.
    points_joined = np.concatenate(point_arrays)
    return points_joined


# İki nokta bulutu arasında dönüşüm matrisini bulmak için fonksiyon.
def find_transform(reference_points, measurement_points):
    """
    Referans nokta bulutu ile ölçüm nokta bulutu arasında dönüşüm bulur.
    
    reference_points: Referans nokta bulutu.
    measurement_points: Ölçülen nokta bulutu.
    """
    # Nokta bulutlarının boyutunu alıyoruz.
    shape = np.shape(reference_points)

    # Referans noktalarını 3D bir yapıya dönüştürüyoruz (z ekseni sıfır olacak şekilde).
    reference_points_3d = np.zeros((shape[0], 3), dtype=np.float32)
    reference_points_3d[:, :2] = reference_points

    # Ölçüm noktalarını da 3D yapıya dönüştürüyoruz.
    measurement_points_3 = np.zeros((measurement_points.shape[0], 3))
    measurement_points_3[:, :2] = measurement_points

    # CPD algoritmasını kullanarak hizalamayı yapıyoruz.
    tf_param, _, _ = cpd.registration_cpd(measurement_points_3, reference_points_3d)

    # Ölçüm noktalarını hizalanmış hale getiriyoruz.
    result = tf_param.transform(measurement_points_3)

    return result, tf_param


# Ana çalışma bloğu.
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # DXF dosyasını nokta bulutuna dönüştürüyoruz.
    points = cad_to_pointcloud("CAM1.dxf")

    # Noktaları görselleştiriyoruz.
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
