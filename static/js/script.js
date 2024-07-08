// 按钮“上传PDF文档”
document.getElementById("genCaseTdd_Upload").addEventListener("click", function () {
  document.getElementById("genCaseTdd_File").click();
});
document.getElementById("genCaseTdd_File").addEventListener("change", function () {
  var fileName = this.value.split("\\").pop();
  document.getElementById("genCaseTdd_FileName").innerText = fileName;
});
document.getElementById("injectKnowledge_Upload").addEventListener("click", function () {
  document.getElementById("injectKnowledge_File").click();
});
document.getElementById("injectKnowledge_File").addEventListener("change", function () {
  var fileName = this.value.split("\\").pop();
  document.getElementById("injectKnowledge_FileName").innerText = fileName;
});

document.getElementById("genCasePrd_Upload").addEventListener("click", function () {
  document.getElementById("genCasePrd_File").click();
});
document.getElementById("genCasePrd_File").addEventListener("change", function () {
  var fileName = this.value.split("\\").pop();
  document.getElementById("genCasePrd_FileName").innerText = fileName;
});

$(document).ready(function () {
  // tab1：表单“仅需求文档”
  $('#genCasePrd').submit(function (e) {
    e.preventDefault();
    var formData = new FormData(this);
    var fileName = document.getElementById("genCasePrd_File").files[0].name; // 获取上传的文件名
    var caseType = $('#genCasePrd_Dropdown').val(); // 获取用户选择的用例类型
    var multimodal = $('#genCasePrd_Dropdown_2').val(); // 获取用户选择的多模态类型
    var fileExtension = caseType === 'excel' ? '.xlsx' : caseType === 'markdown' ? '.md' : '.xmind';

    $('#genCasePrd_LoadingOverlay').show(); // 显示加载动画
    $('#genCasePrd_LoadingSpinner').show();

    $.ajax({
      url: '/genCasePrd',
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      xhrFields: {
        responseType: 'blob'
      },
      success: function (blob) {
        $('#genCasePrd_LoadingOverlay').hide(); // 隐藏加载动画
        $('#genCasePrd_LoadingSpinner').hide();

        // 创建一个指向Blob的URL
        var downloadUrl = URL.createObjectURL(blob);

        // 创建一个临时的不可见<a>标签用于下载
        var a = document.createElement('a');
        a.style.display = 'none';
        a.href = downloadUrl;
        a.download = fileName.slice(0, -4) + fileExtension; // 使用获取到的文件名
        document.body.appendChild(a);
        a.click(); // 模拟点击<a>标签以触发下载

        // 清理：撤销Blob URL并移除<a>标签
        URL.revokeObjectURL(downloadUrl);
        document.body.removeChild(a);
      },
    error: function (xhr) {
        $('#genCasePrd_LoadingOverlay').hide(); // 隐藏加载动画
        $('#genCasePrd_LoadingSpinner').hide();

        var errorMessage = '生成测试用例失败，请重试';
        if (xhr.responseJSON && xhr.responseJSON.error) {
            // 如果错误信息是一个对象，使用JSON.stringify进行转换
            errorMessage = typeof xhr.responseJSON.error === 'object' ? JSON.stringify(xhr.responseJSON.error) : xhr.responseJSON.error;
        }
        alert(errorMessage);
    }
    });
  });
  // tab1：表单“需求+设计文档”
  $('#genCaseTdd').submit(function (e) {
    e.preventDefault();
    var formData = new FormData(this);
    var fileName = document.getElementById("genCaseTdd_File").files[0].name; // 获取上传的文件名
    var caseType = $('#genCaseTdd_Dropdown').val(); // 获取用户选择的用例类型
    var fileExtension = caseType === 'excel' ? '.xlsx' : caseType === 'markdown' ? '.md' : '.xmind';

    $('#genCaseTdd_LoadingOverlay').show(); // 显示加载动画
    $('#genCaseTdd_LoadingSpinner').show();

    $.ajax({
      url: '/genCaseTdd',
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      xhrFields: {
        responseType: 'blob'
      },
      success: function (blob) {
        $('#genCaseTdd_LoadingOverlay').hide(); // 隐藏加载动画
        $('#genCaseTdd_LoadingSpinner').hide();

        // 创建一个指向Blob的URL
        var downloadUrl = URL.createObjectURL(blob);

        // 创建一个临时的不可见<a>标签用于下载
        var a = document.createElement('a');
        a.style.display = 'none';
        a.href = downloadUrl;
        a.download = fileName.slice(0, -4) + fileExtension; // 使用获取到的文件名
        document.body.appendChild(a);
        a.click(); // 模拟点击<a>标签以触发下载

        // 清理：撤销Blob URL并移除<a>标签
        URL.revokeObjectURL(downloadUrl);
        document.body.removeChild(a);
      },
      error: function () {
        $('#genCaseTdd_LoadingOverlay').hide(); // 隐藏加载动画
        $('#genCaseTdd_LoadingSpinner').hide();
        alert('生成测试用例失败，请重试');
      }
    });
  });

  // tab2：表单 注入向量库
  $('#injectKnowledge').submit(function (e) {
    e.preventDefault();
    var formData = new FormData(this);

    $('#injectKnowledge_LoadingOverlay').show();
    $('#injectKnowledge_LoadingSpinner').show();

    $.ajax({
      url: '/injectKnowledge', // Flask后端的URL
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      success: function (response) {
        $('#injectKnowledge_LoadingOverlay').hide(); // 隐藏加载动画
        $('#injectKnowledge_LoadingSpinner').hide();

        if (response.code === 0) {
          $('#ant-popover-inner-success').show();
          setTimeout(function () {
            $('#ant-popover-inner-success').fadeOut(200);
          }, 2000); // 2秒后关闭
        } else if (response.code === 565) {
          alert('该表不存在');
        } else {
          alert('数据获取失败: ' + response.msg);
        }
      },
      error: function () {
        $('#injectKnowledge_LoadingOverlay').hide(); // 隐藏加载动画
        $('#injectKnowledge_LoadingSpinner').hide();
        alert('请求失败，请检查网络和后端服务。');
      }
    });
  });
  
  // tab3：查询按钮
  $('#client_Query').click(function () {
    var spaceName = $('#client_SpaceName').val();
    $.ajax({
      url: '/querySpace', // Flask后端的URL
      type: 'GET',
      data: { spaceName: spaceName },
      success: function (response) {
        if (response.code === 200) {
          updateTable(response.data);
        } else if (response.code === 565) {
          alert('该表不存在');
        } else {
          alert('数据获取失败: ' + response.msg);
        }
      },
      error: function () {
        alert('请求失败，请检查网络和后端服务。');
      }
    });
  });

  // tab3：重置按钮
  $('#client_Reset').click(function () {
    $('#client_SpaceName').val('');
    var spaceName = $('#client_SpaceName').val();
    $.ajax({
      url: '/querySpace', // Flask后端的URL
      type: 'GET',
      data: { spaceName: spaceName },
      success: function (response) {
        if (response.code === 200) {
          updateTable(response.data);
        } else {
          alert('数据获取失败: ' + response.msg);
        }
      },
      error: function () {
        alert('请求失败，请检查网络和后端服务。');
      }
    });
  });

  // tab3：动态更新表格数据
  function updateTable(data) {
    var tbody = $('.ant-table-tbody');
    tbody.empty(); // 清空现有数据
    // 如果data不是数组，将其转换为数组
    if (!Array.isArray(data)) {
      data = [data];
    }
    data.forEach(function (item) {
      var row = `<tr class="ant-table-row ant-table-row-level-0">
        <td class="ant-table-cell">${item.id}</td>
        <td class="ant-table-cell">
          <a class="nereus-link" data-item='${JSON.stringify(item)}' style="display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden;">${item.name}</a>
        </td>
        <td class="ant-table-cell">${item.partition_num}</td>
        <td class="ant-table-cell">${item.replica_num || 'N/A'}</td>
        <td class="ant-table-cell">${item.resource_name || 'N/A'}</td>
      </tr>`;
      tbody.append(row);
    });
  }

  // tab3：显示“创建表空间”弹窗
  $('#client_Create').click(function (e) {
    e.preventDefault(); // 阻止表单默认提交行为
    $('#createSpace_Bg').fadeIn(100); // 显示弹窗
    $('#loading-overlay').fadeIn(100); // 显示背景层
  });

  // tab3：隐藏“创建表空间”弹窗
  $('#createSpace_Close, #createSpace_Close1').click(function (e) {
    e.preventDefault(); // 阻止默认行为
    $('#createSpace_Bg').fadeOut(100); // 隐藏弹窗
    $('#loading-overlay').fadeOut(100); // 隐藏背景层
  });

  // tab3：表单“创建表空间”
  $('#createSpace').submit(function (e) {
    e.preventDefault();
    var spaceName = $('#createSpace_Name').val();
    $('#loadingSpinner').show();
    $.ajax({
      url: '/createSpace',
      type: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ spaceName: spaceName }),
      success: function (response) {
        $('#loadingSpinner').hide();
        if (response.code == 200) {
          $('#ant-popover-inner-success').show();
          setTimeout(function () {
            $('#ant-popover-inner-success').fadeOut(200);
          }, 2000); // 2秒后关闭
          $('#createSpace_Bg').fadeOut(200); // 隐藏弹窗
          $('#loading-overlay').fadeOut(200); // 隐藏背景层
        } else if (response.code === 564) {
          alert('该表已存在');
        } else {
          alert('数据获取失败: ' + response.msg);
        }
      },
      error: function () {
        $('#loadingSpinner').hide();
        alert('请求失败，请检查网络和后端服务。');
      }
    });
  });

  // tab3：“表空间详情”弹窗
  $('.ant-table-tbody').on('click', 'a.nereus-link', function (e) {
    e.preventDefault(); // 阻止表单默认提交行为
    // 获取当前行的数据
    var itemData = $(this).data('item');
    var spaceName = itemData.name;
    // 更新详情弹窗的内容
    $('#detailSpace_Details').find('div').eq(1).text('数据库id: ' + itemData.db_id);
    $('#detailSpace_Details').find('div').eq(2).text('数据表id: ' + itemData.id);
    $('#detailSpace_Details').find('div').eq(3).text('名称: ' + itemData.name);
    $('#detailSpace_Details').find('div').eq(4).text('检索模型: ' + itemData.engine.retrieval_type);
    $('#detailSpace_Details').find('div').eq(5).text('计算方式: ' + itemData.engine.metric_type);
    $('#detailSpace_Details').find('div').eq(6).text('分片索引阀值: ' + itemData.engine.index_size);
    // $('#detailSpace_Details').text(JSON.stringify(itemData, null, 2)); // 格式化显示

    // 表格
    var properties = itemData.properties;
    var tableHTML = '<table><tr>';
    if (properties.hasOwnProperty('feature')) {
      tableHTML += '<th>feature</th>'; // 将feature添加到第一列
    }
    for (var key in properties) {
      if (properties.hasOwnProperty(key) && key !== 'feature') {
        tableHTML += '<th>' + key + '</th>';
      }
    }
    tableHTML += '</tr><tr>';
    if (properties.hasOwnProperty('feature')) {
      tableHTML += '<td>' + (properties.feature.type ? properties.feature.type : '') + '(' + properties.feature.dimension + ')' + '</td>'; // 将feature的类型添加到第一列
    }
    for (var key in properties) {
      if (properties.hasOwnProperty(key) && key !== 'feature') { // 确保feature不被重复处理
        tableHTML += '<td>' + (properties[key].type ? properties[key].type : '') + '</td>'; // 使用<td>（表格单元格）来显示类型
      }
    }
    tableHTML += '</tr></table>';
    $('#detailSpace_Details_Table1').html(tableHTML); // 使用.html()以解析HTML标签

    $('#detailSpace').fadeIn(100); // 显示弹窗
    $('#loading-overlay').fadeIn(100); // 显示背景层

    $('#detailSpace_Close, #detailSpace_Close1').click(function (e) {
      e.preventDefault(); // 阻止默认行为
      $('#detailSpace').fadeOut(100); // 隐藏弹窗
      $('#loading-overlay').fadeOut(100); // 隐藏背景层
    });

    // tab3：“是否删除”弹窗
    $('#detailSpace_Delete').click(function (e) {
      e.preventDefault(); // 阻止表单默认提交行为
      $('#deleteSpace').fadeIn(100); // 显示弹窗
    });
    $('#deleteSpace_Close').click(function (e) {
      e.preventDefault(); // 阻止表单默认提交行为
      $('#deleteSpace').fadeOut(100); // 显示弹窗
    });

    // 当点击“是否删除”弹窗的“确定”按钮时
    $(document).off('click', '#deleteSpace_Commit').on('click', '#deleteSpace_Commit', function (e) {
      e.preventDefault(); // 阻止默认行为
      $.ajax({
        url: '/deleteSpace',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ spaceName: spaceName }), // 发送当前选中项的名称
        success: function (response) {
          if (response.code == 200) {
            $('#ant-popover-inner-success').show();
            $('#deleteSpace').fadeOut(200);
            $('#detailSpace').fadeOut(200);
            $('#loading-overlay').fadeOut(200);
            setTimeout(function () {
              $('#ant-popover-inner-success').fadeOut(100);
            }, 2000); // 2秒后关闭
          } else {
            alert('操作失败: ' + response.msg);
          }
        },
        error: function () {
          alert('请求失败，请检查网络和后端服务。');
        }
      });
    });
  });

});