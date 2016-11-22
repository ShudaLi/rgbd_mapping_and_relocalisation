using namespace std;
using namespace Sophus;


class CMultiViewer : public QGLViewer
{
public:
	CMultiViewer(string strName_, CDataLive::tp_shared_ptr pData, QWidget* parent, const QGLWidget* shareWidget);
    ~CMultiViewer();
protected :
    virtual void draw();
    virtual void init();
    virtual QString helpString() const;
	virtual void keyPressEvent(QKeyEvent *e);
	
	CDataLive::tp_shared_ptr _pData;
	string _strViewer;
	bool _bShowText;
};
